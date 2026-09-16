from fastapi import FastAPI, File, Request, UploadFile
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
async def extract_ocr(file: UploadFile):

    return