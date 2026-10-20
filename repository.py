import uuid

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from config import settings


class VectorRepository:
    """Everything Qdrant-specific lives here. The rest of the app never touches the client."""

    def __init__(self, host: str = settings.qdrant_host, port: int = settings.qdrant_port) -> None:
        self.client = AsyncQdrantClient(host=host, port=port)

    @staticmethod
    def _bm25_document(text: str) -> models.Document:
        """Qdrant/bm25 turns text into a sparse (keyword) vector.
        Stored chunks and queries must be built with the same options."""
        return models.Document(
            text=text,
            model=settings.bm25_model,
            options={"language": settings.bm25_language},
        )

    async def ensure_collection(self, name: str, size: int) -> None:
        """Create the collection only if it doesn't exist. Safe to call on every startup.

        Each point has two named vectors: "dense" (semantic) and "bm25" (keywords).
        The IDF modifier makes Qdrant apply inverse document frequency at query time.
        """
        if await self.client.collection_exists(name):
            return
        logger.debug(f"Creating collection {name}")
        await self.client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(size=size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    async def reset_collection(self, name: str, size: int) -> None:
        """Destructive: deletes the collection (and all its vectors), then recreates it."""
        logger.debug(f"Resetting collection {name}")
        if await self.client.collection_exists(name):
            await self.client.delete_collection(name)
        await self.ensure_collection(name, size)

    async def delete_collection(self, name: str) -> None:
        logger.debug(f"Deleting collection {name}")
        await self.client.delete_collection(name)

    async def upsert_chunks(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict],
        batch_size: int = 100,
    ) -> None:
        """Insert many chunks at once. vectors[i] belongs with payloads[i].
        Every payload must contain "text": it is also used to build the BM25 vector."""
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": vector, "bm25": self._bm25_document(payload["text"])},
                payload=payload,
            )
            for vector, payload in zip(vectors, payloads, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            await self.client.upsert(
                collection_name=collection_name,
                points=points[start : start + batch_size],
            )
        logger.debug(f"Inserted {len(points)} points into {collection_name}")

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        query_text: str,
        limit: int = settings.retrieval_limit,
        score_threshold: float = settings.score_threshold,
        prefetch_limit: int = settings.hybrid_prefetch_limit,
    ) -> list[models.ScoredPoint]:
        """Hybrid search: semantic + keyword candidates, merged with Reciprocal Rank Fusion."""
        response = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=prefetch_limit,
                    score_threshold=score_threshold,  # cosine scores: only meaningful here
                ),
                models.Prefetch(
                    query=self._bm25_document(query_text),
                    using="bm25",
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points


repository = VectorRepository()
