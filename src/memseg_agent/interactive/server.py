"""Loopback-only, single-user workbench. No telemetry or public deployment."""

import argparse
import base64
import binascii
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .session import MAX_ARCHIVE_BYTES, MAX_PIXELS, SegmentationSession, decoded_array, png_bytes


def data_url(array):
    return "data:image/png;base64," + base64.b64encode(png_bytes(array)).decode()


def decode_image(value, *, mask=False):
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("Expected an uploaded PNG/JPEG image")
    payload = base64.b64decode(value.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(payload)) as image:
        if image.format not in ("PNG", "JPEG") or image.width * image.height > MAX_PIXELS:
            raise ValueError("Use PNG/JPEG images with at most 2048×2048 pixels")
        if mask and image.format != "PNG":
            raise ValueError("Masks must be lossless binary PNGs")
        return decoded_array(image, mask=mask)


def create_app(session: SegmentationSession):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response
    from starlette.concurrency import run_in_threadpool
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app = FastAPI(title="MemSeg Workbench", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    assets = Path(__file__).parent / "web"

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.headers.get('host')}"
        if origin and origin != expected:
            return JSONResponse(
                {"detail": "Cross-origin requests are not allowed"}, status_code=403
            )
        if request.url.path.startswith("/api/") and request.headers.get("x-memseg-request") != "1":
            return JSONResponse({"detail": "Local API header required"}, status_code=403)
        response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Cross-Origin-Resource-Policy": "same-origin",
                "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            }
        )
        return response

    async def limited_body(request, limit):
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "Upload exceeds the size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    async def json_body(request):
        try:
            value = json.loads(await limited_body(request, 32 * 1024 * 1024))
            if not isinstance(value, dict):
                raise TypeError()
            return value
        except (ValueError, TypeError, UnicodeDecodeError):
            raise HTTPException(400, "Expected a JSON object")

    def execute(action, body):
        with session.lock:
            if action == "support":
                session.add_support(
                    decode_image(body["image"]),
                    decode_image(body["mask"], mask=True),
                    body.get("name", "Support"),
                )
                return session.summary()
            if action == "query":
                case_id = session.set_query(decode_image(body["image"]), body.get("name", "Query"))
                return {
                    "id": case_id,
                    "image": data_url(session.current["image"]),
                    "shape": list(session.current["image"].shape[:2]),
                }
            if action == "prepare":
                return {"image": data_url(decode_image(body["image"]))}
            if action == "predict":
                mask, retrieved = session.predict(body["id"])
                return {"mask": data_url(mask.astype(np.uint8) * 255), "retrieved": retrieved}
            if action == "correct":
                session.confirm_correction(
                    body["id"],
                    decode_image(body["mask"], mask=True),
                    confirmed=body.get("confirmed", False),
                )
                return session.summary()
            if action == "remove":
                session.remove(body["id"])
                return session.summary()
            if action == "reset":
                if body.get("confirmed") is not True:
                    raise ValueError("Explicit confirmation required to reset")
                session.reset(body.get("target"))
                return session.summary()
            raise ValueError("Unknown action")

    @app.post("/api/{action}")
    async def action_endpoint(action: str, request: Request):
        try:
            if action == "import":
                payload = await limited_body(request, MAX_ARCHIVE_BYTES)
                await run_in_threadpool(session.import_archive, payload)
                return session.summary()
            if action not in (
                "support",
                "query",
                "prepare",
                "predict",
                "correct",
                "remove",
                "reset",
            ):
                raise HTTPException(404, "Unknown action")
            return await run_in_threadpool(execute, action, await json_body(request))
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            binascii.Error,
            UnidentifiedImageError,
            zipfile.BadZipFile,
            OSError,
            Image.DecompressionBombError,
        ) as exc:
            # Do not return paths, tracebacks or uploaded contents in errors.
            detail = str(exc) if isinstance(exc, ValueError) else "Invalid image or memory archive"
            raise HTTPException(400, detail)
        except RuntimeError:
            raise HTTPException(
                503,
                "Model operation failed. Check local console / GPU memory; use fewer retrieved cases or CPU.",
            )

    @app.get("/api/state")
    def state():
        return session.summary()

    @app.get("/api/export")
    def export():
        try:
            payload = session.export_archive()
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return Response(
            payload,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="memseg-reviewed-memories.zip"'},
        )

    @app.get("/api/memory/{memory_id}")
    def memory_detail(memory_id: str):
        with session.lock:
            case = next((c for c in session.cases if c.id == memory_id), None)
            if case is None:
                raise HTTPException(404, "Memory not found")
            return {
                "image": data_url(case.image),
                "mask": data_url(case.mask.astype(np.uint8) * 255),
                "name": case.name,
                "bank": case.bank,
            }

    @app.get("/")
    def index():
        return FileResponse(assets / "index.html")

    @app.get("/app.js")
    def javascript():
        return FileResponse(assets / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def stylesheet():
        return FileResponse(assets / "style.css", media_type="text/css")

    return app


def main():
    parser = argparse.ArgumentParser(
        description="MemSeg local workbench — few-shot + user-corrected working memory"
    )
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--dino-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target", default="Foreground")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not Path(args.sam2_checkpoint).is_file():
        parser.error("SAM2 checkpoint does not exist")
    if not 1 <= args.top_k <= 4 or not 1024 <= args.port <= 65535:
        parser.error("Use top-k 1–4 and port 1024–65535")
    import uvicorn

    from memseg_agent.model import MemSegAgent
    from memseg_agent.utils import resolve_device
    from memseg_agent.working_memory import DINOv3Embedder

    device = resolve_device(args.device)
    print(
        "Loading SAM2 and DINOv3. Only pretrained weights may be downloaded; your cases stay local.",
        flush=True,
    )
    model = (
        MemSegAgent(
            args.sam2_config, args.sam2_checkpoint, memory_mode="none", eval_profile="cross_domain"
        )
        .to(device)
        .eval()
    )
    session = SegmentationSession(
        model, DINOv3Embedder(args.dino_model, device), args.target, args.top_k
    )
    print(
        f"Open http://127.0.0.1:{args.port} — single user, one target per session. Do not expose this service publicly.",
        flush=True,
    )
    uvicorn.run(create_app(session), host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
