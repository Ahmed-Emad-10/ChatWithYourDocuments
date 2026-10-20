import os
import uuid
from contextlib import asynccontextmanager

import aiofiles
from aiofiles.os import makedirs
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from loguru import logger

from config import settings
from llm import generate_answer
from rag import ingest_pdf, retrieve
from repository import repository
from schemas import (
    DocumentStatus,
    GenerateRequest,
    GenerateResponse,
    StatusResponse,
    UploadResponse,
)

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB per read/write

# document_id -> {"filename": str, "status": DocumentStatus, "error": str | None}
# In memory only: resets when the server restarts.
documents: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await repository.ensure_collection(settings.collection_name, settings.embedding_dim)
    yield


app = FastAPI(lifespan=lifespan)


async def save_file(file: UploadFile, document_id: str) -> str:
    filename = os.path.basename(file.filename or "document.pdf")  # strips any path parts
    folder = os.path.join(settings.upload_dir, document_id)
    await makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    async with aiofiles.open(filepath, "wb") as f:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            await f.write(chunk)
    return filepath


async def process_document(document_id: str, filepath: str) -> None:
    """Runs in the background after the upload response has been sent."""
    try:
        count = await ingest_pdf(filepath)
        documents[document_id]["status"] = DocumentStatus.READY
        logger.info(f"Document {document_id} ready ({count} chunks)")
    except Exception as e:
        logger.exception(f"Document {document_id} failed")
        documents[document_id].update(status=DocumentStatus.FAILED, error=str(e))


@app.post("/documents", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="A PDF document"),
    background_tasks: BackgroundTasks = None,
):
    # Check the file's real signature, not the client-supplied content type
    header = await file.read(5)
    await file.seek(0)
    if header != b"%PDF-":
        raise HTTPException(
            detail="Only PDF documents are supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    document_id = str(uuid.uuid4())
    try:
        filepath = await save_file(file, document_id)
    except Exception as e:
        raise HTTPException(
            detail=f"Could not save the file: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    documents[document_id] = {
        "filename": os.path.basename(filepath),
        "status": DocumentStatus.PROCESSING,
        "error": None,
    }
    background_tasks.add_task(process_document, document_id, filepath)
    return UploadResponse(document_id=document_id, filename=documents[document_id]["filename"])


@app.get("/documents/{document_id}/status", response_model=StatusResponse)
async def get_document_status(document_id: str):
    doc = documents.get(document_id)
    if doc is None:
        raise HTTPException(detail="Document not found", status_code=status.HTTP_404_NOT_FOUND)
    return StatusResponse(document_id=document_id, **doc)


@app.post("/generate/text", response_model=GenerateResponse)
async def generate_text_controller(body: GenerateRequest):
    context, sources = await retrieve(body.prompt)
    answer = await generate_answer(body.prompt, context, body.temperature)
    return GenerateResponse(content=answer, sources=sources)
