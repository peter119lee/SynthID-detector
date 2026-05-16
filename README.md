# SynthID / GPT-Image2 Watermark Detector

Detects invisible watermarks in AI-generated images from:
- **Google Gemini Nano Banana Pro** (SynthID)
- **Google Gemini Nano Banana 2** (SynthID)
- **OpenAI GPT-Image2**

Does NOT rely on metadata or EXIF — works on screenshots, re-saved images, and images with stripped metadata.

## How it works

Each AI image model embeds a **fixed noise-level watermark pattern** in every generated image. This pattern is imperceptible to humans but statistically consistent across all images from the same model.

1. **Build phase**: Average the noise residuals of many AI-generated images from the same source. Random content cancels out; the fixed watermark signal reinforces.
2. **Detect phase**: Extract noise residual from a test image (subtract Gaussian-denoised version), then compute normalized correlation with known templates. High correlation = watermark present.

Based on research from [reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Build templates (one-time)

You need 10+ images per AI source. More images = better template.

```bash
python detect.py build \
  --synthid path/to/nano_banana_pro_images/ \
  --gptimage path/to/gpt_image2_images/ \
  --nb2 path/to/nano_banana_2_images/
```

### Step 2: Detect watermarks

```bash
# Basic detection
python detect.py check image.jpg

# Verbose output with all scores
python detect.py check -v image.jpg

# Multiple images
python detect.py check -v img1.png img2.jpg img3.webp

# JSON output
python detect.py check --json image.jpg

# Custom threshold (default: 0.08)
python detect.py check -t 0.05 image.jpg
```

## Output

```
image.jpg: WATERMARK DETECTED - synthid (score=0.283)
photo.png: no watermark detected
```

Exit codes:
- `0` — no watermark detected
- `1` — file error
- `2` — SynthID detected (Nano Banana Pro)
- `3` — GPT-Image2 detected
- `4` — Nano Banana 2 detected

## Performance

- **~46ms** per image (CPU only, no GPU needed)
- Tested accuracy: AI images score 0.15-0.55, real photos score < 0.01
- Dependencies: numpy, Pillow, scipy

## Robustness

| Transformation | Survives? |
|---|---|
| JPEG compression | Yes |
| Resize | Yes (resized to 512x512 internally) |
| Screenshot (no crop) | Yes |
| Format conversion | Yes |
| Metadata stripping | Yes |
| Crop | No |
| Rotation | No |

## Limitations

- **Not 100% accurate** — tested ~98% accuracy. False negatives can occur on heavily compressed images.
- **Heavy JPEG compression kills the watermark** — if the image file is unusually small (e.g. 35KB for a 1024x1024 image), the watermark signal may be too degraded to detect. Check the file size before trusting a "no watermark" result.
- Requires 10+ reference images per AI source to build templates
- Different model versions may need separate templates
- Template quality improves with more reference images (50+ ideal)
- Threshold may need tuning per deployment

## Disclaimer

This is an experimental research tool. It should **NOT** be used as the sole basis for legal decisions, journalism fact-checking, or content moderation. False positives and false negatives can occur. Always verify results with multiple methods.

## TODO

- [ ] Build stronger templates with 200+ images per model (currently 12-14)
- [ ] Test against 50+ diverse real photos (phone cameras, DSLR, memes, screenshots) to validate false positive rate
- [ ] Add support for Grok, Midjourney, DALL-E 3, and other AI image generators
- [ ] Investigate frequency-domain carrier detection for crop robustness

## Files

- `detect.py` — main detector (build + check)
- `templates/` — extracted watermark templates (.npy files)
- `requirements.txt` — Python dependencies
- `gemini-synthid.jpg` — reference fingerprint visualization
- `gptimage2-fingerprint.jpg` — reference fingerprint visualization
