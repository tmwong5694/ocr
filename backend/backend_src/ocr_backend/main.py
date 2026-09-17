from fastapi import BackgroundTasks, FastAPI, Request, UploadFile
from PIL import Image
import pymupdf
import shutil
from pathlib import Path
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


async def _extract_ocr(file: UploadFile):
    pdf_bytes = await file.read()

    prompt_path = Path("ml") / ".config" / "ocr_prompt.txt"
    with open(prompt_path, "r") as f:
        prompt = f.read()

    ocr_model = OCRModel(
        model_name="zai-org/GLM-OCR",
        device="cpu",
        model_kwargs={},
        enable_preprocessing=True,
        max_image_width=768,
        max_image_height=768,
    )
    ocr_model.load_model()

    page_results = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)

            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            page_results.append(
                ocr_model.image_to_text(
                    image=image,
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
):
    return await _extract_ocr(file)