"""Retrieval-Augmented Generation (RAG) pipeline.

A unified, batteries-included RAG pipeline that combines document chunking,
embedding, and context-augmented generation. No external vector database
required — runs entirely locally with Ollama.

Features:
- Hybrid search (semantic + BM25 keyword scoring)
- LLM-based re-ranking of retrieved chunks
- Conversational RAG with history support

Usage::

    from anvil.llm.client import OllamaClient
    from anvil.llm.rag import RAGPipeline

    client = OllamaClient(model="qwen2.5-coder:7b")
    rag = RAGPipeline(client)

    # Ingest documents
    rag.ingest_file("docs/guide.md")
    rag.ingest_text("Python was created by Guido.", source="fact")

    # Query with context-augmented generation
    answer = rag.query("Who created Python?")
    print(answer["response"])   # "Guido van Rossum created Python..."
    print(answer["sources"])    # [SearchResult(doc_id=..., score=0.92)]

    # Conversational RAG — follow-up questions use history
    answer2 = rag.query("When was it released?", history=answer.history)
    # History carries forward so the LLM knows "it" refers to Python

    # Save/load the knowledge base
    rag.save("knowledge.json")
    rag = RAGPipeline.load("knowledge.json", client)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anvil.llm.chunker import chunk_file, chunk_text
from anvil.llm.client import OllamaClient
from anvil.llm.embeddings import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_HYBRID_ALPHA,
    EmbeddingStore,
    SearchResult,
)
from anvil.utils.logging import get_logger

log = get_logger("llm.rag")

__all__ = [
    "RAGPipeline",
    "RAGResult",
    "DEFAULT_RAG_TOP_K",
    "DEFAULT_RAG_CHUNK_SIZE",
    "DEFAULT_RAG_OVERLAP",
    "DEFAULT_RERANK_TOP_N",
    "MAX_CONTEXT_LENGTH",
    "MAX_HISTORY_TURNS",
    "MAX_CONTEXTUALIZE_DOC_CHARS",
    "RAG_SYSTEM_PROMPT",
    "RERANK_PROMPT",
    "CONTEXTUALIZE_PROMPT",
]

# Default RAG parameters
DEFAULT_RAG_TOP_K = 5          # number of chunks to retrieve
DEFAULT_RAG_CHUNK_SIZE = 500   # characters per chunk
DEFAULT_RAG_OVERLAP = 50       # overlap between chunks
DEFAULT_RERANK_TOP_N = 3       # chunks to keep after re-ranking
MAX_CONTEXT_LENGTH = 10_000    # max characters of context to inject
MAX_HISTORY_TURNS = 10         # max conversation turns to carry forward

# System prompt for RAG queries
RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use ONLY the context below to answer. If the context doesn't contain enough "
    "information, say so — do not make up facts.\n\n"
    "Context:\n{context}\n"
)

# Prompt for LLM-based re-ranking
RERANK_PROMPT = (
    "You are a relevance judge. Given a question and a list of text passages, "
    "return a JSON array of passage numbers ranked by relevance to the question. "
    "Most relevant first. Only include passages that are relevant.\n\n"
    "Question: {question}\n\n"
    "Passages:\n{passages}\n\n"
    "Return ONLY a JSON array of passage numbers, e.g. [3, 1, 5]. "
    "Do not include any other text."
)

# Upper bound on how much of the parent document we pass into the
# contextualize prompt. Keeps the call cheap on long documents.
MAX_CONTEXTUALIZE_DOC_CHARS = 40_000

# Contextual-retrieval prompt — Anthropic's pattern. We ask the model to
# situate each chunk within its parent document so the chunk's
# embedding + BM25 tokens carry disambiguating context.
CONTEXTUALIZE_PROMPT = (
    "You are helping to prepare a document chunk for retrieval. "
    "Given the whole document and one specific chunk from it, write ONE short "
    "sentence (no more than 25 words) that situates the chunk in the document. "
    "Mention the surrounding topic, any entity names, and the chunk's role. "
    "Do NOT summarise the chunk itself. Output only the sentence, no preamble, "
    "no bullets, no quotes.\n\n"
    "<document>\n{document}\n</document>\n\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Context sentence:"
)


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Result from a RAG query."""

    response: str
    sources: list[SearchResult]
    context_used: str
    tokens: int
    time_s: float
    history: list[dict[str, str]] = field(default_factory=list)


class RAGPipeline:
    """End-to-end RAG: ingest documents, retrieve context, generate answers.

    Combines the document chunker, embedding store, and LLM client into
    a single pipeline. Supports file and text ingestion, hybrid search
    (semantic + BM25), LLM-based re-ranking, and conversational queries.
    """

    def __init__(
        self,
        client: OllamaClient,
        embed_model: str = DEFAULT_EMBED_MODEL,
        chunk_size: int = DEFAULT_RAG_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_RAG_OVERLAP,
        chunk_strategy: str = "sentence",
        top_k: int = DEFAULT_RAG_TOP_K,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
        rerank: bool = False,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
        contextual_retrieval: bool = False,
        contextualize_prompt: str = CONTEXTUALIZE_PROMPT,
    ):
        self._client = client
        self._store = EmbeddingStore(client, model=embed_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.hybrid_alpha = hybrid_alpha
        self.rerank = rerank
        self.rerank_top_n = rerank_top_n
        # Contextual retrieval (Anthropic pattern): when True, every chunk
        # is passed through ``contextualize_prompt`` alongside its parent
        # document, and the one-sentence context is prepended to the chunk
        # before embedding/indexing. Quality lift comes for free at query
        # time; cost is a one-shot LLM call per chunk during ingestion.
        self.contextual_retrieval = contextual_retrieval
        self.contextualize_prompt = contextualize_prompt

    def __repr__(self) -> str:
        n = len(self._store)
        model = self._store.model
        return f"RAGPipeline(docs={n}, embed_model={model!r})"

    def __len__(self) -> int:
        return len(self._store)

    @property
    def store(self) -> EmbeddingStore:
        """Access the underlying embedding store."""
        return self._store

    def ingest_text(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        contextualize: bool | None = None,
    ) -> int:
        """Chunk and ingest a text string.

        Args:
            text: The text to ingest.
            source: Source identifier (e.g., filename, URL).
            metadata: Additional metadata for each chunk.
            contextualize: Override ``self.contextual_retrieval``. When
                True, each chunk gets a one-sentence context prepended
                before embedding. When None the pipeline default applies.

        Returns the number of chunks successfully ingested.
        """
        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            strategy=self.chunk_strategy,
            source=source,
            metadata=metadata,
        )
        if not chunks:
            return 0

        use_context = self.contextual_retrieval if contextualize is None else contextualize
        if use_context:
            self._contextualize_chunks(chunks, text)

        docs = [
            {"doc_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
            for c in chunks
        ]
        added = self._store.add_batch(docs)
        log.info("Ingested %d/%d chunks from '%s'%s",
                 added, len(chunks), source or "text",
                 " (contextualized)" if use_context else "")
        return added

    def ingest_file(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        contextualize: bool | None = None,
    ) -> int:
        """Chunk and ingest a text file.

        Args:
            path: Path to the file.
            metadata: Additional metadata for each chunk.
            contextualize: Override ``self.contextual_retrieval``. See
                :meth:`ingest_text`.

        Returns the number of chunks successfully ingested.
        """
        chunks = chunk_file(
            path,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            strategy=self.chunk_strategy,
            metadata=metadata,
        )
        if not chunks:
            return 0

        use_context = self.contextual_retrieval if contextualize is None else contextualize
        if use_context:
            # Re-read the file so contextualize sees the whole document.
            try:
                doc_text = Path(path).read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError) as e:
                log.warning("Could not re-read %s for contextualization: %s", path, e)
                doc_text = ""
            if doc_text:
                self._contextualize_chunks(chunks, doc_text)

        docs = [
            {"doc_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
            for c in chunks
        ]
        added = self._store.add_batch(docs)
        log.info("Ingested %d/%d chunks from file '%s'%s",
                 added, len(chunks), path,
                 " (contextualized)" if use_context else "")
        return added

    def _contextualize_chunks(self, chunks: list[Any], document: str) -> None:
        """In-place: prepend a one-sentence context to each chunk's text.

        Applies the Anthropic contextual-retrieval pattern. Each chunk is
        sent to the LLM along with the parent document (truncated to
        :data:`MAX_CONTEXTUALIZE_DOC_CHARS`); the returned sentence is
        prepended to the chunk's text so the embedding + BM25 index
        include the disambiguating context at query time.

        Errors per chunk are isolated: if the LLM call fails, that chunk
        keeps its original text and we move on. Metadata records whether
        contextualization succeeded so callers can tell.
        """
        doc = document[:MAX_CONTEXTUALIZE_DOC_CHARS]
        if len(document) > MAX_CONTEXTUALIZE_DOC_CHARS:
            log.debug("Truncated document to %d chars for contextualization", MAX_CONTEXTUALIZE_DOC_CHARS)

        for chunk in chunks:
            prompt = self.contextualize_prompt.replace("{document}", doc).replace("{chunk}", chunk.text)
            try:
                result = self._client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
            except (OSError, TimeoutError, ValueError) as e:
                log.debug("Contextualize call failed for %s: %s", getattr(chunk, "chunk_id", "?"), e)
                continue
            if result.get("error"):
                continue
            context_line = (result.get("response") or "").strip()
            if not context_line:
                continue
            # Keep it short — if the model ignored the 25-word limit, trim.
            context_line = context_line.splitlines()[0][:400]
            chunk.text = f"{context_line}\n\n{chunk.text}"
            if isinstance(getattr(chunk, "metadata", None), dict):
                chunk.metadata["contextualized"] = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
        hybrid: bool = True,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks for a query without generating a response.

        Args:
            query: The search query text.
            top_k: Number of chunks to retrieve.
            min_score: Minimum similarity score.
            metadata_filter: Filter by metadata key-value pairs.
            hybrid: Use hybrid search (semantic + BM25). Default True.

        Returns list of SearchResult sorted by descending score.
        """
        k = top_k or self.top_k
        if hybrid:
            return self._store.hybrid_search(
                query,
                top_k=k,
                min_score=min_score,
                alpha=self.hybrid_alpha,
                metadata_filter=metadata_filter,
            )
        return self._store.search(
            query,
            top_k=k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )

    def _rerank_results(
        self,
        question: str,
        sources: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Re-rank retrieved chunks using the LLM as a relevance judge.

        Asks the LLM to rank passages by relevance and returns only the
        top-N most relevant, reordered.

        Args:
            question: The user's question.
            sources: Initial retrieval results to re-rank.
            top_n: Number of passages to keep after re-ranking.

        Returns re-ranked and filtered list of SearchResult.
        """
        if len(sources) <= 1:
            return sources

        n = top_n or self.rerank_top_n

        # Build numbered passage list
        passages_text = "\n".join(
            f"[{i + 1}] {src.text[:500]}" for i, src in enumerate(sources)
        )

        prompt = RERANK_PROMPT.replace("{question}", question).replace(
            "{passages}", passages_text
        )

        try:
            result = self._client.chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
                temperature=0.0,
            )
        except (OSError, TimeoutError, ValueError) as e:
            log.warning("Re-ranking LLM call failed, using original order: %s", e)
            return sources[:n]

        if result.get("error"):
            log.warning("Re-ranking LLM error, using original order: %s", result["error"])
            return sources[:n]

        # Parse the ranking from LLM response
        response_text = result.get("response", "")
        try:
            ranking = json.loads(response_text)
            if not isinstance(ranking, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("Could not parse re-ranking response, using original order: %s", e)
            return sources[:n]

        # Map 1-indexed passage numbers back to sources
        reranked: list[SearchResult] = []
        seen: set[int] = set()
        for idx in ranking:
            if not isinstance(idx, int):
                continue
            i = idx - 1  # convert to 0-indexed
            if 0 <= i < len(sources) and i not in seen:
                seen.add(i)
                reranked.append(sources[i])
            if len(reranked) >= n:
                break

        if not reranked:
            log.warning("Re-ranking produced no valid results, using original order")
            return sources[:n]

        log.debug("Re-ranked %d -> %d results", len(sources), len(reranked))
        return reranked

    def query(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        hybrid: bool = True,
        rerank: bool | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> RAGResult:
        """Query with retrieval-augmented generation.

        Retrieves relevant context (using hybrid search by default),
        optionally re-ranks results with the LLM, and generates an answer.
        Supports conversational follow-ups via the history parameter.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            min_score: Minimum similarity score for retrieved chunks.
            metadata_filter: Filter chunks by metadata.
            temperature: Override generation temperature.
            system_prompt: Override system prompt (must contain {context}).
            hybrid: Use hybrid search (semantic + BM25). Default True.
            rerank: Re-rank results with LLM. None uses pipeline default.
            history: Conversation history from a previous RAGResult.history
                     for multi-turn conversations.

        Returns RAGResult with response, sources, history, and metadata.
        """
        start = time.time()

        # Retrieve relevant context
        k = top_k or self.top_k
        use_rerank = rerank if rerank is not None else self.rerank

        # When re-ranking, retrieve more candidates so the re-ranker has a good pool
        retrieve_k = k * 3 if use_rerank else k

        if hybrid:
            sources = self._store.hybrid_search(
                question,
                top_k=retrieve_k,
                min_score=min_score,
                alpha=self.hybrid_alpha,
                metadata_filter=metadata_filter,
            )
        else:
            sources = self._store.search(
                question,
                top_k=retrieve_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
            )

        # Re-rank if enabled
        if use_rerank and len(sources) > 1:
            sources = self._rerank_results(question, sources, top_n=k)

        # Build context string from retrieved chunks
        context_parts = []
        context_len = 0
        for result in sources:
            if context_len + len(result.text) > MAX_CONTEXT_LENGTH:
                break
            context_parts.append(result.text)
            context_len += len(result.text)

        context = "\n---\n".join(context_parts) if context_parts else "No relevant context found."

        # Build system prompt with context
        prompt = (system_prompt or self.system_prompt).replace("{context}", context)

        # Build messages — include conversation history for multi-turn
        messages: list[dict[str, str]] = []
        if history:
            # Trim to max history turns (each turn = user + assistant = 2 messages)
            trimmed = history[-(MAX_HISTORY_TURNS * 2):]
            messages.extend(trimmed)
        messages.append({"role": "user", "content": question})

        # Generate answer
        try:
            llm_result = self._client.chat(
                messages=messages,
                system=prompt,
                temperature=temperature,
            )
        except (OSError, TimeoutError, ValueError) as e:
            log.error("RAG query LLM call failed: %s", e)
            elapsed = time.time() - start
            return RAGResult(
                response=f"Error: LLM call failed ({e})",
                sources=sources,
                context_used=context,
                tokens=0,
                time_s=round(elapsed, 2),
                history=messages,
            )

        elapsed = time.time() - start

        # Check for LLM-level errors
        if "error" in llm_result:
            log.warning("RAG query LLM returned error: %s", llm_result["error"])
            return RAGResult(
                response=f"Error: {llm_result['error']}",
                sources=sources,
                context_used=context,
                tokens=0,
                time_s=round(elapsed, 2),
                history=messages,
            )

        response_text = llm_result.get("response", "")

        # Build updated history for follow-up queries
        updated_history = list(messages)
        updated_history.append({"role": "assistant", "content": response_text})

        return RAGResult(
            response=response_text,
            sources=sources,
            context_used=context,
            tokens=llm_result.get("tokens", 0),
            time_s=round(elapsed, 2),
            history=updated_history,
        )

    def save(self, path: str | Path) -> None:
        """Save the knowledge base to disk."""
        self._store.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        client: OllamaClient,
        **kwargs: Any,
    ) -> RAGPipeline:
        """Load a knowledge base from disk.

        Args:
            path: Path to the saved store file.
            client: OllamaClient for generation and future embeddings.
            **kwargs: Additional RAGPipeline constructor arguments.
        """
        store = EmbeddingStore.load(path, client)
        pipeline = cls(client, embed_model=store.model, **kwargs)
        pipeline._store = store
        return pipeline

    def stats(self) -> dict[str, Any]:
        """Return pipeline statistics."""
        store_stats = self._store.stats()
        return {
            **store_stats,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_strategy": self.chunk_strategy,
            "top_k": self.top_k,
            "hybrid_alpha": self.hybrid_alpha,
            "rerank": self.rerank,
        }

    def clear(self) -> None:
        """Remove all ingested documents."""
        self._store.clear()
