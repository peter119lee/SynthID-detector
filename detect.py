"""
SynthID / GPT-Image2 / Nano Banana Watermark Detector

Detects invisible watermarks in AI-generated images from:
- Google Gemini (Nano Banana Pro / SynthID)
- Google Gemini (Nano Banana 2)
- OpenAI GPT-Image2

Method:
  Each AI model embeds a fixed noise-level watermark pattern in every image.
  By averaging the noise residuals of many images from the same model, we
  extract this pattern as a "template". To detect, we extract the noise
  residual of a test image and correlate it with known templates.

  This works even after JPEG compression, resize, screenshot, and metadata
  stripping — because the watermark lives in the pixel noise, not metadata.

Usage:
    # Step 1: Build templates (need 10+ images per AI source)
    python detect.py build --synthid <folder> --gptimage <folder> --nb2 <folder>

    # Step 2: Detect
    python detect.py check image.jpg
    python detect.py check -v img1.png img2.jpg img3.webp
    python detect.py check --json image.jpg
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


SCRIPT_DIR = Path(__file__).parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"
SIZE = 512  # normalize all images to this size for correlation


def extract_noise(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Extract noise residual by subtracting Gaussian-denoised version."""
    denoised = np.stack([
        gaussian_filter(img[:, :, c].astype(np.float32), sigma)
        for c in range(3)
    ], axis=2)
    return img.astype(np.float32) - denoised


def load_and_resize(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS))


def correlate(residual: np.ndarray, template: np.ndarray) -> float:
    """Max normalized correlation across RGB channels."""
    best = 0.0
    for c in range(3):
        r = residual[:, :, c].ravel()
        t = template[:, :, c].ravel()
        r = r - r.mean()
        t = t - t.mean()
        denom = np.sqrt(np.dot(r, r) * np.dot(t, t))
        if denom > 1e-10:
            best = max(best, float(np.dot(r, t) / denom))
    return best


def build_template(folder: Path) -> np.ndarray:
    """Average noise residuals of all images in folder to extract watermark template."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in extensions]
    if not files:
        print(f"  ERROR: No images found in {folder}", file=sys.stderr)
        sys.exit(1)

    noise_sum = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    for f in files:
        noise_sum += extract_noise(load_and_resize(f))

    return (noise_sum / len(files)).astype(np.float32)


def load_templates() -> dict:
    """Load all saved templates."""
    templates = {}
    if not TEMPLATE_DIR.exists():
        return templates
    for f in TEMPLATE_DIR.glob("*.npy"):
        templates[f.stem] = np.load(f)
    return templates


def build(args):
    """Build watermark templates from AI-generated images."""
    TEMPLATE_DIR.mkdir(exist_ok=True)

    sources = []
    if args.synthid:
        sources.append(("synthid", args.synthid))
    if args.gptimage:
        sources.append(("gptimage2", args.gptimage))
    if args.nb2:
        sources.append(("nanobanana2", args.nb2))

    for name, folder in sources:
        folder = Path(folder)
        n_files = len([f for f in folder.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])
        print(f"Building '{name}' template from {folder} ({n_files} images)...")
        if n_files < 5:
            print(f"  WARNING: Only {n_files} images. 10+ recommended for accuracy.")
        t = build_template(folder)
        np.save(TEMPLATE_DIR / f"{name}.npy", t)
        print(f"  Done. Signal strength: {t.std():.4f}")

    print(f"\nTemplates saved to: {TEMPLATE_DIR}/")
    print("Run: python detect.py check <image>")


def check(args):
    """Check images for watermarks."""
    templates = load_templates()
    if not templates:
        print("ERROR: No templates found. Run 'build' first.", file=sys.stderr)
        print("  python detect.py build --synthid <folder> --gptimage <folder>")
        sys.exit(1)

    threshold = args.threshold if args.threshold else 0.08
    all_results = {}

    for image_path in args.images:
        if not Path(image_path).exists():
            print(f"ERROR: {image_path} not found", file=sys.stderr)
            continue

        img = load_and_resize(Path(image_path))
        residual = extract_noise(img)

        scores = {name: correlate(residual, tmpl) for name, tmpl in templates.items()}
        detected = {n: s for n, s in scores.items() if s > threshold}
        all_results[image_path] = scores

        if args.verbose:
            print(f"Image: {image_path}")
            for name, score in sorted(scores.items(), key=lambda x: -x[1]):
                flag = " << DETECTED" if score > threshold else ""
                print(f"  {name}: {score:.4f}{flag}")
        elif not args.json:
            if detected:
                best = max(detected, key=detected.get)
                print(f"{image_path}: WATERMARK DETECTED - {best} (score={detected[best]:.3f})")
            else:
                print(f"{image_path}: no watermark detected")

    if args.json:
        import json
        print(json.dumps(all_results, indent=2))

    # Exit code per source
    EXIT_CODES = {"synthid": 2, "gptimage2": 3, "nanobanana2": 4}
    for scores in all_results.values():
        for name, s in scores.items():
            if s > threshold:
                sys.exit(EXIT_CODES.get(name, 2))


def main():
    parser = argparse.ArgumentParser(
        description="Detect AI watermarks (SynthID, GPT-Image2, Nano Banana)"
    )
    sub = parser.add_subparsers(dest="command")

    b = sub.add_parser("build", help="Build templates from AI-generated images")
    b.add_argument("--synthid", type=str, help="Folder with Nano Banana Pro images")
    b.add_argument("--gptimage", type=str, help="Folder with GPT-Image2 images")
    b.add_argument("--nb2", type=str, help="Folder with Nano Banana 2 images")

    c = sub.add_parser("check", help="Check images for watermarks")
    c.add_argument("images", nargs="+", help="Image file(s) to check")
    c.add_argument("-v", "--verbose", action="store_true")
    c.add_argument("--json", action="store_true", help="Output JSON")
    c.add_argument("-t", "--threshold", type=float, default=0.08,
                   help="Detection threshold (default: 0.08)")

    args = parser.parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "check":
        check(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
