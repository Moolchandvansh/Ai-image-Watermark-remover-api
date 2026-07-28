import base64
import io
import os
import uuid
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import InferenceClient
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

import huggingface_hub
print(f"huggingface_hub version: {huggingface_hub.__version__}")

# --- TEMPORARY DIAGNOSTIC PATCH ---
# Logs the raw fal.ai response before the library tries to parse it, so we can
# see exactly what came back when the expected "images" key is missing.
try:
    from huggingface_hub.inference._providers import fal_ai as _fal_ai_module
    from huggingface_hub.inference._common import _as_dict as _hf_as_dict

    _original_get_response = _fal_ai_module.FalAIImageToImageTask.get_response

    def _patched_get_response(self, response, request_params=None):
        output = _fal_ai_module.FalAIQueueTask.get_response(self, response, request_params)
        print(f"RAW FAL.AI RESPONSE: {_hf_as_dict(output)}")
        return _original_get_response(self, response, request_params)

    _fal_ai_module.FalAIImageToImageTask.get_response = _patched_get_response
    print("Diagnostic patch applied to FalAIImageToImageTask.get_response")
except Exception as _patch_err:
    print(f"Could not apply diagnostic patch: {_patch_err}")
# --- END TEMPORARY DIAGNOSTIC PATCH ---

HF_TOKEN = os.environ.get("HF_TOKEN")
INFERENCE_PROVIDER = os.environ.get("HF_PROVIDER", "fal-ai")  # fal-ai, replicate, etc.
MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-dev")
OUTPUT_DIR = "/tmp/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not HF_TOKEN:
    # Fail loudly at startup rather than on first request
    print("WARNING: HF_TOKEN environment variable is not set. Requests will fail.")

client: Optional[InferenceClient] = None
if HF_TOKEN:
    client = InferenceClient(provider=INFERENCE_PROVIDER, api_key=HF_TOKEN)

app = FastAPI(
    title="Image Edit API",
    description="Free image editing API powered by Qwen-Image-Edit via Hugging Face Inference Providers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images statically so other projects can just hit a URL
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class EditRequest(BaseModel):
    prompt: str = Field(..., description="Text instruction describing the edit, e.g. 'make the sky sunset orange'")
    image_url: Optional[str] = Field(None, description="Publicly accessible URL of the source image")
    image_base64: Optional[str] = Field(
        None, description="Base64-encoded source image (raw base64 or data URI, e.g. 'data:image/png;base64,...')"
    )


class EditResponse(BaseModel):
    success: bool
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_bytes(req: EditRequest) -> bytes:
    if req.image_base64:
        data = req.image_base64
        if "," in data and data.strip().startswith("data:"):
            data = data.split(",", 1)[1]
        try:
            return base64.b64decode(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    if req.image_url:
        try:
            resp = requests.get(req.image_url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image_url: {e}")

    raise HTTPException(status_code=400, detail="Provide either image_url or image_base64.")


def save_output_image(img: Image.Image) -> str:
    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, format="PNG")
    return filename


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "image-edit-api",
        "model": MODEL_ID,
        "provider": INFERENCE_PROVIDER,
        "endpoints": {
            "edit_post": "POST /edit  (JSON body: prompt, image_url or image_base64)",
            "edit_get": "GET /edit?image_url=...&prompt=...  (redirects to result image)",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "hf_token_configured": bool(HF_TOKEN)}


def run_edit(prompt: str, image_url: Optional[str] = None, image_base64: Optional[str] = None) -> EditResponse:
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Server is not configured with HF_TOKEN. Set it in the environment.",
        )

    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required.")

    req = EditRequest(prompt=prompt, image_url=image_url, image_base64=image_base64)
    img_bytes = load_image_bytes(req)

    try:
        # Normalize/validate the image, and re-encode to PNG bytes for the API
        pil_in = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        pil_in.save(buf, format="PNG")
        clean_bytes = buf.getvalue()

        result = client.image_to_image(
            clean_bytes,
            prompt=prompt.strip(),
            model=MODEL_ID,
        )
    except Exception as e:
        import traceback
        print(f"FULL TRACEBACK for image_to_image error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=502, detail=f"Inference provider error: {type(e).__name__}: {e}")

    # result is typically a PIL.Image
    if isinstance(result, Image.Image):
        out_img = result
    else:
        try:
            out_img = Image.open(io.BytesIO(result))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Unexpected response from provider: {e}")

    filename = save_output_image(out_img)

    out_buf = io.BytesIO()
    out_img.save(out_buf, format="PNG")
    b64_out = base64.b64encode(out_buf.getvalue()).decode("utf-8")

    return EditResponse(
        success=True,
        image_url=f"/outputs/{filename}",
        image_base64=f"data:image/png;base64,{b64_out}",
        message="Edit successful. image_url is relative to this API's base URL.",
    )


@app.post("/edit", response_model=EditResponse)
def edit_image_post(req: EditRequest):
    return run_edit(prompt=req.prompt, image_url=req.image_url, image_base64=req.image_base64)


@app.get("/edit")
def edit_image_get(
    image_url: str,
    prompt: str = "enhance and improve this image",
    redirect: bool = True,
):
    """
    GET-friendly version for pasting directly into a browser or calling from
    tools that only support simple URL requests, e.g.:

    /edit?image_url=https://example.com/photo.jpg&prompt=remove+the+watermark

    By default this redirects straight to the resulting image (302), so the
    URL can be dropped directly into an <img src="..."> tag. Pass
    redirect=false to get the full JSON response instead.
    """
    result = run_edit(prompt=prompt, image_url=image_url)

    if redirect:
        return RedirectResponse(url=result.image_url)

    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
