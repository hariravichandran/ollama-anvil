"""Local embedding store for retrieval-augmented generation (RAG).

A lightweight, dependency-free vector store that uses Ollama embeddings
for semantic search over documents. No external vector database required —
everything runs locally and persists to a single JSON file.

Usage::

    from anvil.llm.client import OllamaClient
    from anvil.llm.embeddings import EmbeddingStore

    client = OllamaClient()
    store = EmbeddingStore(client, model="nomic-embed-text")

    # Add documents
    store.add("doc1", "Python is a programming language.")
    store.add("doc2", "Rust is a systems programming language.")

    # Semantic search
    results = store.search("what language is good for systems?", top_k=3)
    # => [SearchResult(doc_id="doc2", score=0.87, text="Rust is...")]

    # Persist and reload
    store.save("my_store.json")
    store = EmbeddingStore.load("my_store.json", client)
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anvil.llm.client import OllamaClient
from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

log = get_logger("llm.embeddings")

__all__ = [
    "EmbeddingStore",
    "SearchResult",
    "Document",
    "DEFAULT_EMBED_MODEL",
    "MAX_STORE_DOCUMENTS",
    "MAX_TEXT_LENGTH",
    "MAX_METADATA_SIZE",
    "EMBED_BATCH_SIZE",
    "DEFAULT_BM25_K1",
    "DEFAULT_BM25_B",
    "DEFAULT_HYBRID_ALPHA",
]

# Default embedding model — small, fast, good quality
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Store limits
MAX_STORE_DOCUMENTS = 50_000  # max documents in a single store
MAX_TEXT_LENGTH = 50_000      # max characters per document text
MAX_METADATA_SIZE = 10_000    # max characters for serialized metadata

# Embedding batch size — how many texts to embed per API call
EMBED_BATCH_SIZE = 64

# BM25 parameters
DEFAULT_BM25_K1 = 1.5   # term frequency saturation
DEFAULT_BM25_B = 0.75    # length normalization
DEFAULT_HYBRID_ALPHA = 0.7  # weight for semantic score (1-alpha for BM25)


@dataclass(slots=True)
class Document:
    """A document stored in the embedding store."""

    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A search result with similarity score."""

    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingStore:
    """In-memory vector store backed by Ollama embeddings.

    Stores documents with their embeddings and supports cosine similarity
    search. Persists to JSON for simple, portable storage.

    Thread-safe for concurrent reads and writes.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = DEFAULT_EMBED_MODEL,
    ):
        self._client = client
        self.model = model
        self._documents: dict[str, Document] = {}
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        n = len(self._documents)
        dim = self._embedding_dim()
        return f"EmbeddingStore(model={self.model!r}, docs={n}, dim={dim})"

    def __len__(self) -> int:
        return len(self._documents)

    def _embedding_dim(self) -> int:
        """Return the dimension of stored embeddings, or 0 if empty."""
        if not self._documents:
            return 0
        first = next(iter(self._documents.values()))
        return len(first.embedding)

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add a document to the store, computing its embedding.

        If a document with the same doc_id exists, it is replaced.

        Args:
            doc_id: Unique identifier for the document.
            text: The document text to embed and store.
            metadata: Optional metadata dict (e.g., source, tags).

        Returns True on success, False on embedding failure.
        """
        if not doc_id or not text:
            log.warning("Cannot add document with empty id or text")
            return False

        if len(text) > MAX_TEXT_LENGTH:
            log.warning("Truncating document %s text from %d to %d chars", doc_id, len(text), MAX_TEXT_LENGTH)
            text = text[:MAX_TEXT_LENGTH]

        if metadata:
            meta_str = json.dumps(metadata, default=str)
            if len(meta_str) > MAX_METADATA_SIZE:
                log.warning("Metadata too large for %s (%d chars), dropping", doc_id, len(meta_str))
                metadata = {}

        with self._lock:
            if len(self._documents) >= MAX_STORE_DOCUMENTS and doc_id not in self._documents:
                log.error("Store is full (%d documents), cannot add %s", MAX_STORE_DOCUMENTS, doc_id)
                return False

        result = self._client.embed(text, model=self.model)
        embeddings = result.get("embeddings") or []
        if result.get("error") or not embeddings:
            log.error("Failed to embed document %s: %s", doc_id, result.get("error", "no embeddings returned"))
            return False

        embedding = embeddings[0]

        doc = Document(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            created_at=time.time(),
        )

        with self._lock:
            self._documents[doc_id] = doc

        log.debug("Added document %s (dim=%d)", doc_id, len(embedding))
        return True

    def add_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> int:
        """Add multiple documents efficiently using batched embeddings.

        Args:
            documents: List of dicts with keys: doc_id, text, metadata (optional).

        Returns the number of successfully added documents.
        """
        if not documents:
            return 0

        # Validate and prepare
        valid = []
        for doc in documents:
            doc_id = doc.get("doc_id", "")
            text = doc.get("text", "")
            if not doc_id or not text:
                continue
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
            valid.append({"doc_id": doc_id, "text": text, "metadata": doc.get("metadata", {})})

        if not valid:
            return 0

        added = 0
        # Process in batches
        for i in range(0, len(valid), EMBED_BATCH_SIZE):
            batch = valid[i:i + EMBED_BATCH_SIZE]
            texts = [d["text"] for d in batch]

            result = self._client.embed(texts, model=self.model)
            if result.get("error") or not result.get("embeddings"):
                log.error("Batch embed failed: %s", result.get("error", "no embeddings"))
                continue

            embeddings = result["embeddings"]
            if len(embeddings) != len(batch):
                log.error("Embedding count mismatch: got %d, expected %d", len(embeddings), len(batch))
                continue

            now = time.time()
            with self._lock:
                for doc_info, embedding in zip(batch, embeddings, strict=False):
                    if len(self._documents) >= MAX_STORE_DOCUMENTS and doc_info["doc_id"] not in self._documents:
                        log.warning("Store full, stopping batch add at %d docs", added)
                        return added
                    self._documents[doc_info["doc_id"]] = Document(
                        doc_id=doc_info["doc_id"],
                        text=doc_info["text"],
                        embedding=embedding,
                        metadata=doc_info.get("metadata", {}),
                        created_at=now,
                    )
                    added += 1

        log.info("Batch added %d/%d documents", added, len(valid))
        return added

    def remove(self, doc_id: str) -> bool:
        """Remove a document by ID. Returns True if it existed."""
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                return True
        return False

    def get(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID."""
        return self._documents.get(doc_id)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Semantic search: find documents most similar to the query.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            min_score: Minimum cosine similarity score (0.0 to 1.0).
            metadata_filter: If provided, only return documents whose
                metadata contains all specified key-value pairs.

        Returns list of SearchResult sorted by descending similarity.
        """
        if not query or not self._documents:
            return []

        result = self._client.embed(query, model=self.model)
        embeddings = result.get("embeddings") or []
        if result.get("error") or not embeddings:
            log.error("Failed to embed query: %s", result.get("error", "no embeddings returned"))
            return []

        query_vec = embeddings[0]

        scored: list[tuple[str, float]] = []
        with self._lock:
            for doc_id, doc in self._documents.items():
                # Apply metadata filter
                if metadata_filter and not all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

                score = _cosine_similarity(query_vec, doc.embedding)
                if score >= min_score:
                    scored.append((doc_id, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored[:top_k]:
            doc = self._documents[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                text=doc.text,
                score=round(score, 4),
                metadata=doc.metadata,
            ))

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining semantic similarity and BM25 keyword scoring.

        Fuses cosine similarity (semantic) with BM25 (keyword) scores using
        weighted combination. This improves recall for queries that contain
        exact terms (names, identifiers) that pure semantic search may miss.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            min_score: Minimum combined score (0.0 to 1.0).
            alpha: Weight for semantic score. BM25 weight is (1 - alpha).
                   Default 0.7 favors semantic but boosts keyword matches.
            metadata_filter: Filter chunks by metadata key-value pairs.

        Returns list of SearchResult sorted by descending combined score.
        """
        if not query or not self._documents:
            return []

        # Get semantic embeddings
        result = self._client.embed(query, model=self.model)
        embeddings = result.get("embeddings") or []
        if result.get("error") or not embeddings:
            log.error("Failed to embed query: %s", result.get("error", "no embeddings"))
            return []

        query_vec = embeddings[0]

        # Build BM25 index and score
        bm25 = _BM25Index()
        with self._lock:
            bm25.build(self._documents)

        bm25_scores = bm25.score(query)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        if max_bm25 > 0:
            bm25_norm = {k: v / max_bm25 for k, v in bm25_scores.items()}
        else:
            bm25_norm = bm25_scores

        # Compute hybrid scores
        scored: list[tuple[str, float]] = []
        with self._lock:
            for doc_id, doc in self._documents.items():
                if metadata_filter and not all(
                    doc.metadata.get(k) == v for k, v in metadata_filter.items()
                ):
                    continue

                sem_score = _cosine_similarity(query_vec, doc.embedding)
                kw_score = bm25_norm.get(doc_id, 0.0)
                combined = alpha * sem_score + (1 - alpha) * kw_score

                if combined >= min_score:
                    scored.append((doc_id, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored[:top_k]:
            doc = self._documents[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                text=doc.text,
                score=round(score, 4),
                metadata=doc.metadata,
            ))

        return results

    def list_documents(self) -> list[str]:
        """Return all document IDs."""
        return list(self._documents.keys())

    def clear(self) -> None:
        """Remove all documents."""
        with self._lock:
            self._documents.clear()

    def save(self, path: str | Path) -> None:
        """Save the store to a JSON file."""
        path = Path(path)
        with self._lock:
            data = {
                "model": self.model,
                "version": 1,
                "documents": [
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "embedding": doc.embedding,
                        "metadata": doc.metadata,
                        "created_at": doc.created_at,
                    }
                    for doc in self._documents.values()
                ],
            }
        atomic_write(path, json.dumps(data))
        log.info("Saved embedding store to %s (%d documents)", path, len(data["documents"]))

    @classmethod
    def load(cls, path: str | Path, client: OllamaClient) -> EmbeddingStore:
        """Load a store from a JSON file.

        Args:
            path: Path to the saved store file.
            client: OllamaClient instance for future embedding calls.

        Returns a new EmbeddingStore populated with the saved documents.
        """
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        model = raw.get("model", DEFAULT_EMBED_MODEL)
        store = cls(client=client, model=model)

        for doc_data in raw.get("documents", []):
            doc = Document(
                doc_id=doc_data["doc_id"],
                text=doc_data["text"],
                embedding=doc_data["embedding"],
                metadata=doc_data.get("metadata", {}),
                created_at=doc_data.get("created_at", 0.0),
            )
            store._documents[doc.doc_id] = doc

        log.info("Loaded embedding store from %s (%d documents, model=%s)", path, len(store), model)
        return store

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        with self._lock:
            if not self._documents:
                return {"documents": 0, "model": self.model, "dimension": 0}
            first = next(iter(self._documents.values()))
            return {
                "documents": len(self._documents),
                "model": self.model,
                "dimension": len(first.embedding),
            }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Ollama embeddings are L2-normalized, so this simplifies to dot product.
    But we compute the full formula for safety with non-normalized vectors.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class _BM25Index:
    """Lightweight BM25 index for keyword-based scoring.

    Operates over the same document set as EmbeddingStore. Rebuilt on
    demand from the current documents — not persisted separately.
    """

    def __init__(self, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B):
        self.k1 = k1
        self.b = b
        # doc_id -> {term: count}
        self._tf: dict[str, dict[str, int]] = {}
        # doc_id -> doc length in tokens
        self._doc_len: dict[str, int] = {}
        # term -> set of doc_ids containing it
        self._df: dict[str, set[str]] = {}
        self._avg_dl: float = 0.0
        self._n: int = 0

    def build(self, documents: dict[str, Document]) -> None:
        """Rebuild the index from a document dict."""
        self._tf.clear()
        self._doc_len.clear()
        self._df.clear()

        total_len = 0
        for doc_id, doc in documents.items():
            tokens = _tokenize(doc.text)
            self._doc_len[doc_id] = len(tokens)
            total_len += len(tokens)

            tf: dict[str, int] = {}
            seen: set[str] = set()
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
                if token not in seen:
                    seen.add(token)
                    if token not in self._df:
                        self._df[token] = set()
                    self._df[token].add(doc_id)
            self._tf[doc_id] = tf

        self._n = len(documents)
        self._avg_dl = total_len / max(1, self._n)

    def score(self, query: str, doc_ids: list[str] | None = None) -> dict[str, float]:
        """Score documents against a query. Returns {doc_id: bm25_score}."""
        if self._n == 0:
            return {}

        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}

        targets = doc_ids if doc_ids is not None else list(self._tf.keys())
        scores: dict[str, float] = {}

        for doc_id in targets:
            tf = self._tf.get(doc_id)
            if tf is None:
                continue
            dl = self._doc_len.get(doc_id, 0)
            s = 0.0
            for term in query_tokens:
                df = len(self._df.get(term, set()))
                if df == 0:
                    continue
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
                term_tf = tf.get(term, 0)
                numerator = term_tf * (self.k1 + 1)
                denominator = term_tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                s += idf * numerator / denominator
            scores[doc_id] = s

        return scores
