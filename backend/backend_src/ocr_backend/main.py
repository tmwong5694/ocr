from fastapi import FastAPI, File, Request, UploadFile
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
@app.post("/analyze_file/", tags=["files"])
async def analyze_file(file: UploadFile, request: Request):
    return await _analyze_file_stat(file, request)

@app.post("/files/ocr", tags=["files"])
async def extract_ocr(
        file: UploadFile,

):
    model_name = "zai-org/GLM-OCR"
    temp_path = ""
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
        max_width=768,
        max_height=768,
    )
    ocr_model.image_to_text(
        image_path=temp_path,
        prompt=prompt,
        model_name=model_name
    )
    return