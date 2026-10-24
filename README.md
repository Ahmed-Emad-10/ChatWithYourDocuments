# Chat with your PDF

A retrieval-augmented generation (RAG) app. Upload PDF documents, ask questions in a chat UI, and get answers grounded in your documents, with the filename and page number of every source.

## Features

- **Hybrid search** in Qdrant: multilingual dense embeddings (`BAAI/bge-m3`) combined with BM25 keyword vectors, merged with Reciprocal Rank Fusion (RRF)
- **Page-aware chunking**: fixed-size chunks with overlap, each stored with its source file and page number
- **Async FastAPI backend**: streamed uploads and background ingestion
- **Document status tracking**: `processing`, `ready` or `failed`, so clients know when a document is searchable
- **GPU support**: embeddings run on CUDA (fp16) or Apple Silicon (MPS) when available, otherwise on CPU
- **Streamlit chat UI** with live document status and source citations
- **Qdrant in Docker Compose**: one command to start the vector database
- **Provider-agnostic LLM**: answers are generated through LangChain, so you choose the model

## Architecture

```
Ingestion (background task)
  PDF upload -> extract text per page -> chunk -> embed (bge-m3) + BM25 vectors -> Qdrant

Query (live)
  Question -> embed -> Qdrant hybrid search (dense + BM25, RRF) -> top chunks -> LLM -> answer + sources
```

## Project structure

```
.
├── main.py              # FastAPI app and endpoints
├── config.py            # Settings (overridable via .env)
├── schemas.py           # Pydantic request/response models
├── rag.py               # Extraction, chunking, embedding, ingestion, retrieval
├── repository.py        # Qdrant wrapper (collection, upsert, hybrid search)
├── llm.py               # LLM call through LangChain (you configure the provider)
├── ui.py                # Streamlit chat interface
├── docker-compose.yml   # Qdrant container
├── requirements.txt
├── uploads/             # Saved PDFs (created at runtime, gitignored)
└── qdrant_storage/      # Qdrant data (created by Docker, gitignored)
```

## Requirements

- Python 3.10+
- Docker and Docker Compose
- An API key (or local model) for the LLM provider you choose
- Optional: an NVIDIA GPU for much faster embedding. Install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) **before** the other dependencies, otherwise you get the CPU-only build.

## Setup

1. **Start Qdrant**

   ```bash
   docker compose up -d
   ```

   The dashboard is at http://localhost:6333/dashboard.

2. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Then install the LangChain package for your LLM provider (for example `langchain-openai`, `langchain-anthropic` or `langchain-ollama`).

3. **Configure your LLM**

   Open `llm.py` and fill in `build_llm()` so it returns your LangChain chat model. Put your API key in `.env` or in the environment.

4. **Run the API**

   ```bash
   uvicorn main:app --reload
   ```

   The first start downloads the embedding model (about 2 GB), so it takes a while.

5. **Run the UI** (in a second terminal)

   ```bash
   streamlit run ui.py
   ```

   Open http://localhost:8501. If your API is not on port 8000, change `API_URL` at the top of `ui.py`.

## Usage

In the UI, upload a PDF from the sidebar, wait until its status shows ✅, then ask questions in the chat. Each answer has a "Sources" section listing the file and page of the chunks it used.

### API

Interactive docs are at http://localhost:8000/docs.

**Upload a document**

```bash
curl -X POST http://localhost:8000/documents -F "file=@document.pdf"
# {"document_id": "…", "filename": "document.pdf", "status": "processing"}
```

**Check its status**

```bash
curl http://localhost:8000/documents/<document_id>/status
# {"document_id": "…", "filename": "document.pdf", "status": "ready", "error": null}
```

**Ask a question**

```bash
curl -X POST http://localhost:8000/generate/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is this document about?", "temperature": 0.7}'
# {"content": "…", "sources": [{"source": "document.pdf", "page": 3, "score": 0.03}]}
```

## Configuration

All settings live in `config.py` and can be overridden with environment variables or a `.env` file (for example `CHUNK_SIZE=800`).

| Setting | Default | Description |
|---|---|---|
| `qdrant_host` / `qdrant_port` | `localhost` / `6333` | Where Qdrant is running |
| `collection_name` | `knowledgebase` | Qdrant collection |
| `embedding_model` | `BAAI/bge-m3` | Sentence-transformers model |
| `embedding_dim` | `1024` | Must match the embedding model's output size |
| `chunk_size` / `chunk_overlap` | `512` / `64` | Chunking, in characters |
| `retrieval_limit` | `3` | Chunks passed to the LLM |
| `score_threshold` | `0.7` | Minimum cosine score for the dense (semantic) search |
| `bm25_language` | `english` | Stemming and stopwords for BM25 (for example `arabic`) |
| `hybrid_prefetch_limit` | `20` | Candidates taken from each search before fusion |
| `upload_dir` | `uploads` | Where PDFs are saved |

## Notes and limitations

- **Changing the embedding model, its dimension, or the collection layout requires a reset.** Run `await repository.reset_collection(settings.collection_name, settings.embedding_dim)` (for example in a notebook) and upload your documents again.
- **Document status is kept in memory.** It resets when the API restarts. The vectors in Qdrant are not affected.
- **Scanned PDFs are not supported.** There is no OCR, so a PDF without a text layer is marked `failed`.
- **Arabic PDFs:** `pypdf` can struggle with Arabic text, so check the extraction quality on your files. Set `bm25_language` to match your documents.
- **Scores:** the `score` returned with each source is a fused RRF rank score, not a similarity percentage. `score_threshold` only applies to the semantic half of the search.
- **BM25 matches any shared word**, so a question with a common term can still return chunks even when nothing is truly relevant.
- **Chunking is fixed-size and per page**, so a sentence that crosses a page break is split.

## Tech stack

Python, FastAPI, Qdrant, sentence-transformers (BGE-M3), FastEmbed (BM25), LangChain, pypdf, Streamlit, Docker Compose
