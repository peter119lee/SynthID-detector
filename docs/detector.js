/**
 * AI Watermark Detector - Client-side detection of SynthID / GPT-Image2 watermarks
 * Crop-robust via multiscale sliding correlation.
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
 * Correlate residual (tw x th) against a sub-region of template at offset (oy, ox)
 */
function corrPatch(residual, tw, th, template, oy, ox, c) {
  const rn = tw * th;
  const rOff = c * rn;
  let sumR = 0;
  for (let i = 0; i < rn; i++) sumR += residual[rOff + i];
  const meanR = sumR / rn;

  let dot = 0, normR = 0, normT = 0;
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const ri = rOff + y * tw + x;
      const ti = c * SIZE * SIZE + (y + oy) * SIZE + (x + ox);
      const r = residual[ri] - meanR;
      const t = template[ti]; // template already zero-mean per channel from build
      dot += r * t;
      normR += r * r;
      normT += t * t;
    }
  }
  const denom = Math.sqrt(normR * normT);
  return denom > 1e-10 ? dot / denom : 0;
}

/**
 * Strategy 1: stretch to 512x512 (fast, for full images)
 */
async function scoreStretch(file, template) {
  const rgb = await loadImageAtSize(file, SIZE, SIZE);
  const n = SIZE * SIZE;
  let best = 0;
  for (let c = 0; c < 3; c++) {
    const ch = rgb.subarray(c * n, (c + 1) * n);
    const blurred = gaussianBlur(ch, SIZE, SIZE, 2.0);
    const residual = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) residual[c * n + i] = ch[i] - blurred[i];
    // Direct full correlation
    let sumR = 0, sumT = 0;
    for (let i = 0; i < n; i++) { sumR += residual[c * n + i]; sumT += template[c * n + i]; }
    const meanR = sumR / n, meanT = sumT / n;
    let dot = 0, nR = 0, nT = 0;
    for (let i = 0; i < n; i++) {
      const r = residual[c * n + i] - meanR;
      const t = template[c * n + i] - meanT;
      dot += r * t; nR += r * r; nT += t * t;
    }
    const denom = Math.sqrt(nR * nT);
    if (denom > 1e-10) best = Math.max(best, dot / denom);
  }
  return best;
}

/**
 * Strategy 2: multiscale sliding (for crops)
 * Try different scale assumptions and slide over template to find best match.
 */
async function scoreMultiscale(file, template) {
  const ratios = [0.85, 0.75, 0.6, 0.5];
  let best = 0;

  for (const ratio of ratios) {
    const tw = Math.round(SIZE * ratio);
    const th = Math.round(SIZE * ratio);
    const rgb = await loadImageAtSize(file, tw, th);
    const rn = tw * th;

    // Extract residual
    const residual = new Float32Array(rn * 3);
    for (let c = 0; c < 3; c++) {
      const ch = rgb.subarray(c * rn, (c + 1) * rn);
      const blurred = gaussianBlur(ch, tw, th, 2.0);
      for (let i = 0; i < rn; i++) residual[c * rn + i] = ch[i] - blurred[i];
    }

    // Slide over template with coarse step
    const maxOY = SIZE - th;
    const maxOX = SIZE - tw;
    const step = Math.max(1, Math.round(maxOY / 4));

    for (let oy = 0; oy <= maxOY; oy += step) {
      for (let ox = 0; ox <= maxOX; ox += step) {
        for (let c = 0; c < 3; c++) {
          const s = corrPatch(residual, tw, th, template, oy, ox, c);
          if (s > best) best = s;
        }
      }
    }
  }
  return best;
}

/**
 * Main detection: stretch first (fast), multiscale fallback if not detected
 */
async function detect(file) {
  const tmpl = await loadTemplates();
  const scores = {};

  // Fast path: stretch
  for (const [name, template] of Object.entries(tmpl)) {
    scores[name] = await scoreStretch(file, template);
  }

  if (Math.max(...Object.values(scores)) > THRESHOLD) {
    let detected = null, maxScore = 0;
    for (const [name, score] of Object.entries(scores)) {
      if (score > THRESHOLD && score > maxScore) { detected = name; maxScore = score; }
    }
    return { scores, detected, maxScore };
  }

  // Slow path: multiscale sliding for crops
  for (const [name, template] of Object.entries(tmpl)) {
    const s2 = await scoreMultiscale(file, template);
    if (s2 > scores[name]) scores[name] = s2;
  }

  let detected = null, maxScore = 0;
  for (const [name, score] of Object.entries(scores)) {
    if (score > THRESHOLD && score > maxScore) { detected = name; maxScore = score; }
  }
  return { scores, detected, maxScore };
}
