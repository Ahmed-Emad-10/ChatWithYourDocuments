from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable values live here. Override any of them via .env or environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "knowledgebase"

    # Embedding model (embedding_dim is fixed by the model and must match the collection)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Chunking (measured in characters for now)
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    retrieval_limit: int = 3
    score_threshold: float = 0.3

    # Hybrid search (dense + BM25, fused with RRF)
    bm25_model: str = "Qdrant/bm25"
    bm25_language: str = "english"
    hybrid_prefetch_limit: int = 20  # candidates taken from each search before fusion

    # File storage
    upload_dir: str = "uploads"


settings = Settings()
