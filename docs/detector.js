/**
 * AI Watermark Detector - Client-side detection of SynthID / GPT-Image2 watermarks
 * Uses Gaussian blur + normalized correlation with pre-extracted templates.
 */

const SIZE = 512;
const THRESHOLD = 0.08;

let templates = null; // { name: Float32Array[512*512*3] }

/**
 * Load and decompress all templates
 */
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
    for (let i = 0; i < int8.length; i++) {
      float32[i] = int8[i] * scale;
    }
    templates[name] = float32;
  }
  return templates;
}

/**
 * Decompress gzipped data using DecompressionStream API
 */
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
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result.buffer;
}

/**
 * Load image from File/Blob, resize to 512x512, return RGB float32 array
 */
function loadImage(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = SIZE;
      canvas.height = SIZE;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, SIZE, SIZE);
      const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
      // Convert RGBA to RGB float32 (planar: all R, then all G, then all B)
      const pixels = imageData.data;
      const rgb = new Float32Array(SIZE * SIZE * 3);
      const n = SIZE * SIZE;
      for (let i = 0; i < n; i++) {
        rgb[i] = pixels[i * 4];           // R
        rgb[n + i] = pixels[i * 4 + 1];   // G
        rgb[2 * n + i] = pixels[i * 4 + 2]; // B
      }
      resolve(rgb);
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  });
}

/**
 * Apply Gaussian blur to a single channel (512x512)
 * Uses separable 2-pass box blur approximation (3 passes = good Gaussian approx)
 */
function gaussianBlur(channel, sigma) {
  const w = SIZE, h = SIZE;
  // Box blur radius for 3-pass approximation of Gaussian
  const radius = Math.round(sigma * 1.5) | 0;
  const out = new Float32Array(w * h);
  const tmp = new Float32Array(w * h);

  function boxBlurH(src, dst, r) {
    const iarr = 1 / (r + r + 1);
    for (let y = 0; y < h; y++) {
      let ti = y * w, li = ti, ri = ti + r;
      let val = src[ti] * (r + 1);
      for (let j = 0; j < r; j++) val += src[ti + j];
      for (let j = 0; j <= r; j++) { val += src[ri++] - src[ti]; dst[ti++] = val * iarr; }
      for (let j = r + 1; j < w - r; j++) { val += src[ri++] - src[li++]; dst[ti++] = val * iarr; }
      for (let j = w - r; j < w; j++) { val += src[ri - 1] - src[li++]; dst[ti++] = val * iarr; }
    }
  }

  function boxBlurV(src, dst, r) {
    const iarr = 1 / (r + r + 1);
    for (let x = 0; x < w; x++) {
      let ti = x, li = ti, ri = ti + r * w;
      let val = src[ti] * (r + 1);
      for (let j = 0; j < r; j++) val += src[ti + j * w];
      for (let j = 0; j <= r; j++) { val += src[ri] - src[ti]; dst[ti] = val * iarr; ti += w; ri += w; }
      for (let j = r + 1; j < h - r; j++) { val += src[ri] - src[li]; dst[ti] = val * iarr; li += w; ti += w; ri += w; }
      for (let j = h - r; j < h; j++) { val += src[ri - w] - src[li]; dst[ti] = val * iarr; li += w; ti += w; }
    }
  }

  // 3-pass box blur approximates Gaussian
  out.set(channel);
  for (let pass = 0; pass < 3; pass++) {
    boxBlurH(out, tmp, radius);
    boxBlurV(tmp, out, radius);
  }
  return out;
}

/**
 * Extract noise residual: image - gaussian_blur(image)
 */
function extractNoise(rgb) {
  const n = SIZE * SIZE;
  const residual = new Float32Array(n * 3);
  for (let c = 0; c < 3; c++) {
    const channel = rgb.subarray(c * n, (c + 1) * n);
    const blurred = gaussianBlur(channel, 2.0);
    for (let i = 0; i < n; i++) {
      residual[c * n + i] = channel[i] - blurred[i];
    }
  }
  return residual;
}

/**
 * Normalized correlation between residual and template (max across channels)
 */
function correlate(residual, template) {
  const n = SIZE * SIZE;
  let best = 0;
  for (let c = 0; c < 3; c++) {
    const offset = c * n;
    let sumR = 0, sumT = 0;
    for (let i = 0; i < n; i++) {
      sumR += residual[offset + i];
      sumT += template[offset + i];
    }
    const meanR = sumR / n, meanT = sumT / n;

    let dot = 0, normR = 0, normT = 0;
    for (let i = 0; i < n; i++) {
      const r = residual[offset + i] - meanR;
      const t = template[offset + i] - meanT;
      dot += r * t;
      normR += r * r;
      normT += t * t;
    }
    const denom = Math.sqrt(normR * normT);
    if (denom > 1e-10) {
      best = Math.max(best, dot / denom);
    }
  }
  return best;
}

/**
 * Main detection function
 * @param {File} file - Image file
 * @returns {Promise<{scores: Object, detected: string|null, maxScore: number}>}
 */
async function detect(file) {
  const tmpl = await loadTemplates();
  const rgb = await loadImage(file);
  const residual = extractNoise(rgb);

  const scores = {};
  for (const [name, template] of Object.entries(tmpl)) {
    scores[name] = correlate(residual, template);
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
