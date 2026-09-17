from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
import pymupdf
import shutil
import tempfile
from pathlib import Path
from typing import Literal
from ml_pipeline.ocr_model import OCRModel

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "success", "message": "API is online"}

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


async def _extract_ocr(
        file: UploadFile,
        background_tasks: BackgroundTasks
):
    # Check file type
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=415,
            detail="Only .pdf files are accepted.",
        )


    # 1. Create a unique temporary directory
    temp_dir = tempfile.mkdtemp()

    # 2. Schedule cleanup AFTER FastAPI returns the HTTP response
    background_tasks.add_task(cleanup_temp_dir, temp_dir)

    pdf_path = Path(temp_dir) / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model_name = "zai-org/GLM-OCR"
    # temp_path = ""
    prompt_path = Path("ml") / ".config" / "ocr_prompt.txt"

    try:
        with open(prompt_path, "r") as f:
            prompt = f.read()
    except FileNotFoundError:
        return {"error": f"Prompt file not found at {prompt_path}"}

    ocr_model = OCRModel(
        model_name=model_name,
        device="cpu",
        model_kwargs={},
        enable_preprocessing=True,
        max_image_width=768,
        max_image_height=768,
    )
    ocr_model.load_model()

    page_results = []
    with pymupdf.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image_path = Path(temp_dir) / f"page_{page_index + 1:04d}.png"
            # pix.save(image_path) # TODO: not saving yet

            page_results.append(
                ocr_model.image_to_text(
                    image_path=str(image_path),
                    prompt=prompt,
                    model_name="zai-org/GLM-OCR",
                )
            )
    return {"pages": page_results, "text": "\n\n".join(page_results)}


@app.post("/analyze_file/", tags=["files"])
async def analyze_file(file: UploadFile, request: Request):
    return await _analyze_file_stat(file, request)

@app.post("/files/ocr", tags=["files"])
async def extract_ocr(
        file: UploadFile,
        background_tasks: BackgroundTasks
):
    return await _extract_ocr(file, background_tasks)