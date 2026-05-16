/**
 * AI Watermark Detector - Client-side detection of SynthID / GPT-Image2 watermarks
 * Dual strategy: full-image stretch + proportional sub-region for crop robustness.
 */

const SIZE = 512;
const THRESHOLD = 0.08;

let templates = null;

async function loadTemplates() {
  if (templates) return templates;
  const metaResp = await fetch('templates.json');
  const meta = await metaResp.json();
  templates = {};
  for (const [name, info] of Object.entries(meta)) {
    const resp = await fetch(`${name}.bin`);
    const compressed = await resp.arrayBuffer();
    const raw = await decompress(compressed);
    const int8 = new Int8Array(raw);
    const float32 = new Float32Array(int8.length);
    const scale = info.scale / 127;
    for (let i = 0; i < int8.length; i++) float32[i] = int8[i] * scale;
    templates[name] = float32;
  }
  return templates;
}

async function decompress(buffer) {
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(new Uint8Array(buffer));
  writer.close();
  const reader = ds.readable.getReader();
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
  return result.buffer;
}

/**
 * Load image at exact target size, return planar RGB Float32Array
 */
function loadImageAtSize(file, w, h) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);
      const pixels = imageData.data;
      const n = w * h;
      const rgb = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        rgb[i] = pixels[i * 4];
        rgb[n + i] = pixels[i * 4 + 1];
        rgb[2 * n + i] = pixels[i * 4 + 2];
      }
      resolve(rgb);
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  });
}

/**
 * 3-pass box blur approximating Gaussian for arbitrary dimensions
 */
function gaussianBlur(channel, w, h, sigma) {
  const n = w * h;
  const radius = Math.round(sigma * 1.5) | 0;
  if (radius < 1) return new Float32Array(channel);
  const out = new Float32Array(n);
  const tmp = new Float32Array(n);

  function boxBlurH(src, dst, r) {
    const iarr = 1 / (r + r + 1);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let ti = row, li = row, ri = row + r;
      let val = src[row] * (r + 1);
      for (let j = 0; j < r; j++) val += src[row + Math.min(j, w - 1)];
      for (let j = 0; j <= r; j++) { val += src[Math.min(ri++, row + w - 1)] - src[row]; dst[ti++] = val * iarr; }
      for (let j = r + 1; j < w - r; j++) { val += src[ri++] - src[li++]; dst[ti++] = val * iarr; }
      for (let j = w - r; j < w; j++) { val += src[row + w - 1] - src[li++]; dst[ti++] = val * iarr; }
    }
  }

  function boxBlurV(src, dst, r) {
    const iarr = 1 / (r + r + 1);
    for (let x = 0; x < w; x++) {
      let ti = x, li = x, ri = x + r * w;
      let val = src[x] * (r + 1);
      for (let j = 0; j < r; j++) val += src[x + Math.min(j, h - 1) * w];
      for (let j = 0; j <= r; j++) { val += src[Math.min(ri, x + (h-1)*w)] - src[x]; dst[ti] = val * iarr; ti += w; ri += w; }
      for (let j = r + 1; j < h - r; j++) { val += src[ri] - src[li]; dst[ti] = val * iarr; li += w; ti += w; ri += w; }
      for (let j = h - r; j < h; j++) { val += src[x + (h-1)*w] - src[li]; dst[ti] = val * iarr; li += w; ti += w; }
    }
  }

  out.set(channel);
  for (let pass = 0; pass < 3; pass++) {
    boxBlurH(out, tmp, radius);
    boxBlurV(tmp, out, radius);
  }
  return out;
}

/**
 * Normalized correlation between two flat arrays
 */
function normCorr(a, b) {
  const n = a.length;
  let sumA = 0, sumB = 0;
  for (let i = 0; i < n; i++) { sumA += a[i]; sumB += b[i]; }
  const meanA = sumA / n, meanB = sumB / n;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    const va = a[i] - meanA, vb = b[i] - meanB;
    dot += va * vb; na += va * va; nb += vb * vb;
  }
  const denom = Math.sqrt(na * nb);
  return denom > 1e-10 ? dot / denom : 0;
}

/**
 * Strategy 1: Stretch to 512x512 (works for full/near-full images)
 */
async function scoreStretch(file, template) {
  const rgb = await loadImageAtSize(file, SIZE, SIZE);
  const n = SIZE * SIZE;
  let best = 0;
  for (let c = 0; c < 3; c++) {
    const ch = rgb.subarray(c * n, (c + 1) * n);
    const blurred = gaussianBlur(ch, SIZE, SIZE, 2.0);
    const residual = new Float32Array(n);
    for (let i = 0; i < n; i++) residual[i] = ch[i] - blurred[i];
    const tCh = template.subarray(c * n, (c + 1) * n);
    best = Math.max(best, normCorr(residual, tCh));
  }
  return best;
}

/**
 * Strategy 2: Proportional sub-region (works for crops)
 * Assumes crop is from center of original. Tries multiple sub-region sizes.
 */
async function scoreSubRegion(file, template) {
  // Try the image at several proportional sizes within 512x512
  const ratios = [1.0, 0.75, 0.5];
  let best = 0;

  for (const ratio of ratios) {
    const tw = Math.round(SIZE * ratio);
    const th = Math.round(SIZE * ratio);
    if (tw < 64 || th < 64) continue;

    const rgb = await loadImageAtSize(file, tw, th);
    const n = tw * th;

    // Template sub-region (centered)
    const tl = Math.round((SIZE - tw) / 2);
    const tt = Math.round((SIZE - th) / 2);

    for (let c = 0; c < 3; c++) {
      const ch = rgb.subarray(c * n, (c + 1) * n);
      const blurred = gaussianBlur(ch, tw, th, 2.0);
      const residual = new Float32Array(n);
      for (let i = 0; i < n; i++) residual[i] = ch[i] - blurred[i];

      // Extract template sub-region
      const tSub = new Float32Array(n);
      for (let y = 0; y < th; y++) {
        for (let x = 0; x < tw; x++) {
          tSub[y * tw + x] = template[c * SIZE * SIZE + (y + tt) * SIZE + (x + tl)];
        }
      }
      best = Math.max(best, normCorr(residual, tSub));
    }
  }
  return best;
}

/**
 * Main detection: run both strategies, take max score
 */
async function detect(file) {
  const tmpl = await loadTemplates();
  const scores = {};

  for (const [name, template] of Object.entries(tmpl)) {
    const s1 = await scoreStretch(file, template);
    const s2 = await scoreSubRegion(file, template);
    scores[name] = Math.max(s1, s2);
  }

  let detected = null;
  let maxScore = 0;
  for (const [name, score] of Object.entries(scores)) {
    if (score > THRESHOLD && score > maxScore) {
      detected = name;
      maxScore = score;
    }
  }

  return { scores, detected, maxScore };
}
