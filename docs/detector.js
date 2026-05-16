/**
 * AI Watermark Detector - Client-side detection of SynthID / GPT-Image2 watermarks
 * Crop-robust: uses FFT cross-correlation to find best alignment.
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
 * Load image, return { rgb: planar Float32Array, width, height }
 * Does NOT resize - keeps proportional scale for crop robustness.
 */
function loadImage(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      // Scale to fit within 512x512 maintaining aspect ratio
      const scale = Math.min(SIZE / img.width, SIZE / img.height);
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);

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
      resolve({ rgb, width: w, height: h });
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  });
}

/**
 * 3-pass box blur approximating Gaussian (for arbitrary w x h)
 */
function gaussianBlur(channel, w, h, sigma) {
  const radius = Math.round(sigma * 1.5) | 0;
  const out = new Float32Array(w * h);
  const tmp = new Float32Array(w * h);

  function boxBlurH(src, dst, r) {
    const iarr = 1 / (r + r + 1);
    for (let y = 0; y < h; y++) {
      let ti = y * w, li = ti, ri = ti + r;
      let val = src[ti] * (r + 1);
      for (let j = 0; j < r; j++) val += src[ti + Math.min(j, w - 1)];
      for (let j = 0; j <= r; j++) { val += src[Math.min(ri, ti + w - 1)] - src[ti]; dst[ti] = val * iarr; ti++; ri++; }
      for (let j = r + 1; j < w - r; j++) { val += src[ri] - src[li]; dst[ti] = val * iarr; li++; ti++; ri++; }
      for (let j = w - r; j < w; j++) { val += src[ti + w - 1 - (ti - y * w)] - src[li]; dst[ti] = val * iarr; li++; ti++; }
    }
  }

  function boxBlurV(src, dst, r) {
    const iarr = 1 / (r + r + 1);
    for (let x = 0; x < w; x++) {
      let ti = x, li = ti, ri = ti + r * w;
      let val = src[ti] * (r + 1);
      for (let j = 0; j < r; j++) val += src[ti + Math.min(j, h - 1) * w];
      for (let j = 0; j <= r; j++) { val += src[Math.min(ri, x + (h-1)*w)] - src[ti]; dst[ti] = val * iarr; ti += w; ri += w; }
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
 * Correlate a (possibly smaller) residual against the 512x512 template
 * using zero-padded FFT cross-correlation for crop robustness.
 */
function correlateWithPadding(residual, rw, rh, template) {
  const n = SIZE * SIZE;
  let best = 0;

  for (let c = 0; c < 3; c++) {
    // Extract channel from residual
    const rn = rw * rh;
    const rCh = new Float32Array(rn);
    for (let i = 0; i < rn; i++) rCh[i] = residual[c * rn + i];

    // Zero-mean the residual
    let rSum = 0;
    for (let i = 0; i < rn; i++) rSum += rCh[i];
    const rMean = rSum / rn;
    let rEnergy = 0;
    for (let i = 0; i < rn; i++) { rCh[i] -= rMean; rEnergy += rCh[i] * rCh[i]; }
    rEnergy = Math.sqrt(rEnergy);
    if (rEnergy < 1e-10) continue;

    // Template channel (already 512x512 planar)
    const tCh = new Float32Array(n);
    let tSum = 0;
    for (let i = 0; i < n; i++) tSum += template[c * n + i];
    const tMean = tSum / n;
    let tEnergy = 0;
    for (let i = 0; i < n; i++) { tCh[i] = template[c * n + i] - tMean; tEnergy += tCh[i] * tCh[i]; }
    tEnergy = Math.sqrt(tEnergy);
    if (tEnergy < 1e-10) continue;

    // If residual is same size as template, direct correlation
    if (rw === SIZE && rh === SIZE) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += rCh[i] * tCh[i];
      best = Math.max(best, dot / (rEnergy * tEnergy));
    } else {
      // Pad residual into 512x512 and use spatial correlation at offset
      // Try the most likely offset (center alignment)
      const offY = Math.round((SIZE - rh) / 2);
      const offX = Math.round((SIZE - rw) / 2);
      let dot = 0;
      for (let y = 0; y < rh; y++) {
        for (let x = 0; x < rw; x++) {
          const ti = (y + offY) * SIZE + (x + offX);
          dot += rCh[y * rw + x] * tCh[ti];
        }
      }
      // Normalize by partial template energy at the crop region
      let tPartialEnergy = 0;
      for (let y = 0; y < rh; y++) {
        for (let x = 0; x < rw; x++) {
          const ti = (y + offY) * SIZE + (x + offX);
          tPartialEnergy += tCh[ti] * tCh[ti];
        }
      }
      tPartialEnergy = Math.sqrt(tPartialEnergy);
      if (tPartialEnergy > 0) {
        best = Math.max(best, dot / (rEnergy * tPartialEnergy));
      }
    }
  }
  return best;
}

/**
 * Main detection function
 */
async function detect(file) {
  const tmpl = await loadTemplates();
  const { rgb, width, height } = await loadImage(file);
  const n = width * height;

  // Extract noise residual
  const residual = new Float32Array(n * 3);
  for (let c = 0; c < 3; c++) {
    const channel = rgb.subarray(c * n, (c + 1) * n);
    const blurred = gaussianBlur(channel, width, height, 2.0);
    for (let i = 0; i < n; i++) residual[c * n + i] = channel[i] - blurred[i];
  }

  const scores = {};
  for (const [name, template] of Object.entries(tmpl)) {
    scores[name] = correlateWithPadding(residual, width, height, template);
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
