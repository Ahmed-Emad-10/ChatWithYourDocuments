import asyncio
import os
import re

import torch
from loguru import logger
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import settings
from repository import repository
from schemas import Source

# Loaded once at import time: slow to start, but every request reuses it.
def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = _pick_device()
embedder = SentenceTransformer(settings.embedding_model, device=device)
if device == "cuda":
    embedder.half()  # fp16: faster and uses about half the GPU memory
logger.info(f"Embedding model loaded on {device}")


# ---------- 1. Extraction (blocking: pypdf has no async API) ----------


def extract_pages(filepath: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with page numbers starting at 1. Skips empty pages."""
    reader = PdfReader(filepath)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


# ---------- 2. Chunking (fixed size, with overlap) ----------


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break  # this chunk already reached the end of the text
    return chunks


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """Chunk each page separately so every chunk keeps its page number."""
    chunks = []
    for page_number, text in pages:
        for chunk in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append({"page": page_number, "text": chunk})
    return chunks


# ---------- 3. Embedding (blocking: run in a thread) ----------


def _encode(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts, normalize_embeddings=True).tolist()


async def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(await asyncio.to_thread(_encode, texts[start : start + batch_size]))
    return vectors


async def embed_query(query: str) -> list[float]:
    return (await embed_batch([query]))[0]


# ---------- 4. Ingestion: PDF -> chunks -> vectors -> Qdrant ----------


async def ingest_pdf(filepath: str) -> int:
    """Process one saved PDF and store its chunks. Returns the number of chunks stored."""
    source = os.path.basename(filepath)
    logger.debug(f"Ingesting {source}")

    pages = await asyncio.to_thread(extract_pages, filepath)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No extractable text found (is this a scanned PDF?)")

    vectors = await embed_batch([c["text"] for c in chunks])
    payloads = [{"source": source, **c} for c in chunks]

    await repository.ensure_collection(settings.collection_name, settings.embedding_dim)
    await repository.upsert_chunks(settings.collection_name, vectors, payloads)
    return len(chunks)


# ---------- 5. Retrieval: question -> relevant chunks ----------


async def retrieve(query: str) -> tuple[str, list[Source]]:
    """Return (context_text_for_the_prompt, sources_for_the_response)."""
    query_vector = await embed_query(query)
    points = await repository.search(settings.collection_name, query_vector, query)
    context = "\n\n".join(p.payload["text"] for p in points)
    sources = [
        Source(source=p.payload["source"], page=p.payload["page"], score=p.score)
        for p in points
    ]
    return context, sources
