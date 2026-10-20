from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus = DocumentStatus.PROCESSING


class StatusResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    error: str | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class Source(BaseModel):
    source: str  
    page: int
    score: float


class GenerateResponse(BaseModel):
    content: str
    sources: list[Source] = []
