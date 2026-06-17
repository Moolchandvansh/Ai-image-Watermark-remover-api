"""
watermark_api.py — single-file FastAPI watermark removal API
YOLOv8n watermark detection + LaMa inpainting · URL support · no limits

Install:
    pip install fastapi uvicorn[standard] python-multipart pillow numpy \
                torch torchvision huggingface-hub httpx ultralytics

Run:
    python watermark_api.py

Endpoints:
    GET  /health      — model status
    POST /remove/url  — JSON { "image_url": "https://..." }
    GET  /docs        — Swagger UI

How it works:
    1. Download image from URL
    2. YOLOv8n detects text/logo watermark regions → bounding boxes
    3. Draw mask from bounding boxes
    4. LaMa inpaints the masked regions
    5. Return clean image
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import gc
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

# ── third-party ────────────────────────────────────────────────────────────────
import httpx
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("watermark-api")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CACHE_DIR   = os.getenv("MODEL_CACHE_DIR",   "/tmp/watermark_models")
LAMA_REPO_ID      = os.getenv("LAMA_REPO_ID",      "smartywu/big-lama")
LAMA_FILENAME     = os.getenv("LAMA_FILENAME",      "big-lama.pt")
YOLO_REPO_ID      = os.getenv("YOLO_REPO_ID",      "qfisch/yolov8n-watermark-detection")
YOLO_FILENAME     = os.getenv("YOLO_FILENAME",      "best.pt")
PAD_MULTIPLE      = 8
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

_lock        = threading.Lock()
_lama        = None   # TorchScript LaMa
_yolo        = None   # Ultralytics YOLO
_models_ready = False


def _download(repo_id: str, filename: str) -> Path:
    dest = Path(MODEL_CACHE_DIR) / filename
    if dest.exists():
        log.info("Cached: %s", dest)
        return dest
    Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s from %s …", filename, repo_id)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODEL_CACHE_DIR))


def load_models() -> None:
    global _lama, _yolo, _models_ready
    if _models_ready:
        return
    with _lock:
        if _models_ready:
            return

        # ── LaMa ──────────────────────────────────────────────────────────────
        lama_path = _download(LAMA_REPO_ID, LAMA_FILENAME)
        log.info("Loading LaMa …")
        _lama = torch.jit.load(str(lama_path), map_location=DEVICE)
        _lama.eval()
        log.info("✅ LaMa ready")

        # ── YOLOv8n watermark detector ────────────────────────────────────────
        yolo_path = _download(YOLO_REPO_ID, YOLO_FILENAME)
        log.info("Loading YOLOv8n watermark detector …")
        from ultralytics import YOLO
        _yolo = YOLO(str(yolo_path))
        log.info("✅ YOLOv8n ready")

        _models_ready = True
        log.info("🚀 All models loaded on %s", DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION — YOLOv8n finds watermark bounding boxes
# ══════════════════════════════════════════════════════════════════════════════

def detect_watermarks(
    pil_image: Image.Image,
    confidence: float = 0.25,
    padding: int = 10,
) -> list[tuple[int, int, int, int]]:
    """
    Run YOLOv8n on the image.
    Returns list of (x1, y1, x2, y2) bounding boxes in pixel coords.
    padding expands each box slightly to cover soft watermark edges.
    """
    if not _models_ready:
        load_models()

    results = _yolo(pil_image, conf=confidence, verbose=False)
    W, H    = pil_image.size
    boxes   = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # expand box by padding pixels, clamp to image bounds
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(W, int(x2) + padding)
            y2 = min(H, int(y2) + padding)
            boxes.append((x1, y1, x2, y2))
            log.info(
                "Detected watermark @ [%d,%d,%d,%d] conf=%.2f",
                x1, y1, x2, y2, float(box.conf[0]),
            )

    return boxes


def boxes_to_mask(
    boxes: list[tuple[int, int, int, int]],
    image_size: Tuple[int, int],
) -> Image.Image:
    """Draw white rectangles on black canvas for each detected box."""
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box, fill=255)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# INPAINTING — LaMa fills the masked regions
# ══════════════════════════════════════════════════════════════════════════════

def _pad_tensors(img: torch.Tensor, msk: torch.Tensor):
    _, _, H, W = img.shape
    pw = (PAD_MULTIPLE - W % PAD_MULTIPLE) % PAD_MULTIPLE
    ph = (PAD_MULTIPLE - H % PAD_MULTIPLE) % PAD_MULTIPLE
    p  = (0, pw, 0, ph)
    return F.pad(img, p, "reflect"), F.pad(msk, p, "reflect"), H, W


def inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    image : (H, W, 3) uint8 RGB
    mask  : (H, W)    uint8  255=inpaint region  0=keep
    returns (H, W, 3) uint8 RGB
    """
    if not _models_ready:
        load_models()

    img_t = torch.from_numpy(image).float().div(255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    msk_t = torch.from_numpy(mask).float().div(255).unsqueeze(0).unsqueeze(0).to(DEVICE)
    msk_t = (msk_t > 0.5).float()

    img_t, msk_t, H, W = _pad_tensors(img_t, msk_t)

    with torch.inference_mode():
        out = _lama(img_t, msk_t)

    out = out[:, :, :H, :W]
    return out.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def bytes_to_pil(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Unreadable image — send JPEG, PNG, or WEBP.")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return buf.getvalue()


def fmt_from_str(s: str) -> str:
    s = s.upper()
    return s if s in MIME else "PNG"


async def fetch_url(url: str) -> bytes:
    """Download any public image URL (including Telegram file URLs)."""
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(400, f"Failed to fetch URL ({e.response.status_code}): {url}")
    except httpx.RequestError as e:
        raise HTTPException(400, f"Network error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    pil_image: Image.Image,
    confidence: float,
    padding: int,
    output_format: str,
) -> tuple[bytes, int]:
    """
    Detect watermarks → build mask → inpaint → return bytes.
    Returns (image_bytes, num_watermarks_found).
    """
    # 1. Detect
    boxes = detect_watermarks(pil_image, confidence=confidence, padding=padding)

    if not boxes:
        log.info("No watermarks detected — returning original image")
        return pil_to_bytes(pil_image, output_format), 0

    # 2. Build mask from boxes
    mask = boxes_to_mask(boxes, pil_image.size)

    # 3. Inpaint
    result_np = inpaint(np.array(pil_image), np.array(mask))
    result_pil = Image.fromarray(result_np)

    return pil_to_bytes(result_pil, output_format), len(boxes)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class RemoveRequest(BaseModel):
    """
    JSON body for POST /remove/url

    Examples
    --------
    Minimal:
        { "image_url": "https://example.com/photo.jpg" }

    Full:
        {
          "image_url": "https://example.com/photo.jpg",
          "confidence": 0.25,
          "padding": 10,
          "output_format": "PNG"
        }

    Telegram bot:
        { "image_url": "https://api.telegram.org/file/bot<TOKEN>/<path>" }
    """
    image_url:     str
    confidence:    float = 0.25   # YOLOv8 detection threshold (0.1–0.9)
    padding:       int   = 10     # pixels to expand each detected box
    output_format: str   = "PNG"  # PNG | JPEG | WEBP

    @field_validator("confidence")
    @classmethod
    def clamp_conf(cls, v):
        return max(0.05, min(0.95, v))

    @field_validator("padding")
    @classmethod
    def clamp_pad(cls, v):
        return max(0, min(100, v))

    @field_validator("output_format")
    @classmethod
    def validate_fmt(cls, v):
        return fmt_from_str(v)


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Starting — loading models …")
    try:
        load_models()
    except Exception as e:
        log.warning("Model pre-load deferred: %s", e)
    yield
    gc.collect()
    log.info("🛑 Shutdown")


app = FastAPI(
    title="Watermark Removal API",
    description=(
        "Remove watermarks from any hosted image URL.\n\n"
        "**Detection**: YOLOv8n trained specifically on watermarks — finds text and logos.\n\n"
        "**Inpainting**: LaMa fills the detected regions realistically.\n\n"
        "Send any public image URL — works with Telegram, Imgur, Cloudinary, S3, anywhere."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
    <html><head><title>Watermark Removal API</title></head>
    <body style="font-family:monospace;padding:2rem;max-width:560px">
      <h2>🖼️ Watermark Removal API v3</h2>
      <p>YOLOv8n detection + LaMa inpainting</p>
      <pre style="background:#f4f4f4;padding:1rem">
POST /remove/url
Content-Type: application/json

{
  "image_url": "https://example.com/photo.jpg",
  "confidence": 0.25,
  "output_format": "PNG"
}
      </pre>
      <a href="/docs">→ Interactive docs</a>
    </body></html>
    """


@app.get("/health", tags=["System"])
async def health():
    """Check if both models are loaded and ready."""
    return {
        "status": "ok",
        "models_ready": _models_ready,
        "lama_loaded": _lama is not None,
        "yolo_loaded": _yolo is not None,
        "device": DEVICE,
    }


@app.post(
    "/remove/url",
    tags=["Watermark Removal"],
    summary="Remove watermarks from a hosted image URL",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Cleaned image with watermarks removed",
            "headers": {
                "X-Watermarks-Found": {"description": "Number of watermarks detected"},
            },
        },
        400: {"description": "Bad URL or unreadable image"},
        404: {"description": "No watermarks detected"},
    },
)
async def remove_from_url(req: RemoveRequest, background_tasks: BackgroundTasks):
    """
    The main endpoint. Send any public image URL and get back the watermark-free image.

    **Detection model**: `qfisch/yolov8n-watermark-detection` — trained to find
    text watermarks and logo watermarks specifically.

    **Inpainting model**: LaMa (`smartywu/big-lama`) — fills detected regions
    with realistic content using the surrounding image context.

    **Telegram bot usage**:
    ```python
    file = await bot.get_file(message.photo[-1].file_id)
    tg_url = f"https://api.telegram.org/file/bot{TOKEN}/{file.file_path}"
    r = httpx.post("http://your-api/remove/url", json={"image_url": tg_url})
    await message.answer_photo(BufferedInputFile(r.content, "result.png"))
    ```

    **Tune detection**:
    - Lower `confidence` (e.g. 0.1) → catches more watermarks, may have false positives
    - Higher `confidence` (e.g. 0.5) → more precise, may miss faint watermarks
    - Increase `padding` → expands mask around each detected box (good for soft edges)
    """
    # Download image from URL
    image_bytes = await fetch_url(req.image_url)
    try:
        pil_image = bytes_to_pil(image_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))

    log.info("Processing %s | size=%s | conf=%.2f", req.image_url, pil_image.size, req.confidence)

    try:
        out_bytes, n_found = run_pipeline(
            pil_image,
            confidence=req.confidence,
            padding=req.padding,
            output_format=req.output_format,
        )
    except Exception as e:
        log.error("Pipeline error: %s", e)
        raise HTTPException(500, f"Processing failed: {e}")

    background_tasks.add_task(gc.collect)

    return Response(
        content=out_bytes,
        media_type=MIME[req.output_format],
        headers={"X-Watermarks-Found": str(n_found)},
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "watermark_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM BOT  (save as bot.py, run separately)
# ══════════════════════════════════════════════════════════════════════════════
#
# pip install aiogram httpx
#
# import asyncio, os, httpx
# from aiogram import Bot, Dispatcher, Router, F
# from aiogram.types import Message, BufferedInputFile
#
# BOT_TOKEN = os.environ["BOT_TOKEN"]
# API_URL   = os.environ.get("API_URL", "http://localhost:8000")
# bot, dp, router = Bot(BOT_TOKEN), Dispatcher(), Router()
# dp.include_router(router)
#
# @router.message(F.photo)
# async def handle_photo(message: Message):
#     await message.reply("🔍 Detecting watermarks…")
#     file   = await bot.get_file(message.photo[-1].file_id)
#     tg_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"
#     async with httpx.AsyncClient(timeout=120) as client:
#         r = await client.post(f"{API_URL}/remove/url", json={"image_url": tg_url})
#     n = r.headers.get("X-Watermarks-Found", "?")
#     if r.status_code == 200:
#         await message.answer_photo(
#             BufferedInputFile(r.content, "clean.png"),
#             caption=f"✅ Done! {n} watermark(s) removed."
#         )
#     else:
#         await message.reply(f"❌ Error: {r.text}")
#
# @router.message(F.text)
# async def handle_text(msg: Message):
#     await msg.reply("Send me a photo and I'll remove the watermark! 🖼️")
#
# if __name__ == "__main__":
#     asyncio.run(dp.start_polling(bot))
