from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import json
import pymupdf
import shutil
from pathlib import Path
from ml_pipeline.ocr_model import OCRModel

BASE_DIR = Path(__file__).resolve().parents[3]
FRONTEND_DIR = BASE_DIR / "frontend"

@asynccontextmanager
async def lifespan(app: FastAPI):
    prompt_path = BASE_DIR / "ml" / ".config" / "ocr_prompt.txt"
    app.state.prompt = prompt_path.read_text()

    app.state.ocr_model = OCRModel(
        model_name="zai-org/GLM-OCR",
        device="cpu",
        model_kwargs={},
        enable_preprocessing=True,
        max_image_width=768,
        max_image_height=768,
    )
    app.state.ocr_model.load_model()

    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _analyze_file_stat(file: UploadFile, request: Request):
    file_size = request.headers.get("content-length")
    return {
        "filename": file.filename,
        "file_size": int(file_size) if file_size else 0,
        "content_type": file.content_type
    }


def cleanup_temp_dir(temp_dir: str):
    """Deletes the temporary directory and all contents after the response is sent."""
    shutil.rmtree(temp_dir, ignore_errors=True)

def _parse_ocr_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip(" \n\'")
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        parsed = eval(cleaned)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {e.msg}") from e

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Model returned JSON that is not an object.")

    return parsed

async def _extract_ocr(file: UploadFile, request: Request):

    pdf_bytes = await file.read()
    prompt = request.app.state.prompt
    ocr_model = request.app.state.ocr_model

    page_results = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            raw_text = ocr_model.image_to_text(
                image=image,
                prompt=prompt,
                model_name="zai-org/GLM-OCR",
            )
            page_results.append(_parse_ocr_json(raw_text))

    return {
        "pages": page_results,
        "text": page_results[0] if len(page_results) == 1 else page_results,
    }

@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.post("/analyze_file/", tags=["files"])
async def analyze_file(file: UploadFile, request: Request):
    return await _analyze_file_stat(file, request)

@app.post("/files/ocr", tags=["files"])
async def extract_ocr(file: UploadFile, request: Request):
    return await _extract_ocr(file, request)