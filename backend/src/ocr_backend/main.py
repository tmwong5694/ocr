from fastapi import FastAPI, File, Request, UploadFile


app = FastAPI()

@app.get("/")
async def root():
    return {"status": "success", "message": "API is online"}

@app.post("/files/", tags=["files"])
async def create_file(file: UploadFile, request: Request):
    file_size = request.headers.get("content-length")
    return {
        "filename": file.filename,
        "file_size": int(file_size) if file_size else 0,
        "content_type": file.content_type
    }