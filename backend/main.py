from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import AnalyzeResponse
from backend.services.content_router import detect_content_type, route_to_model
from backend.services.decision_engine import finalize_result
from backend.services.explainability import attach_explainability_metadata


app = FastAPI(
    title="EDITH",
    version="1.0.0",
    description="Multi-modal authenticity analysis system for text, image, video, and audio content.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[EDITH] Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "type": "unknown",
            "signal_score": 0.5,
            "confidence": 0.4,
            "status": "Suspicious",
            "explanation": f"Server error: {str(exc)}",
            "evidence": ["Internal server error - please try again"],
            "details": {"error": str(exc)},
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "EDITH is online."}


@app.get("/health")
async def health() -> dict[str, object]:
    import importlib.util

    checks = {}
    for pkg in ["librosa", "cv2", "torch", "transformers", "pydub", "duckduckgo_search"]:
        checks[pkg] = importlib.util.find_spec(pkg) is not None
    return {"status": "ok", "dependencies": checks}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
):
    inline_text = text
    if file is None and inline_text is None:
        try:
            payload = await request.json()
            inline_text = (payload or {}).get("text")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc
        except Exception:
            pass

    if file is None and not inline_text:
        raise HTTPException(status_code=400, detail="Provide either a file or text input.")

    content_type = detect_content_type(
        filename=file.filename if file else None,
        mime_type=file.content_type if file else None,
        text=inline_text,
    )
    if content_type == "unknown":
        raise HTTPException(status_code=400, detail="Unsupported or unknown content type.")

    temp_path: Path | None = None
    try:
        if file is not None:
            suffix = Path(file.filename or "upload.bin").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(await file.read())
                temp_path = Path(temp_file.name)

        raw_result = route_to_model(content_type, text=inline_text, file_path=str(temp_path) if temp_path else None)
        response_payload = finalize_result(content_type, raw_result)
        return attach_explainability_metadata(content_type, response_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
