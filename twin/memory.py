"""A small on-disk vector store. Plain numpy, so there is nothing extra to install.

Each chunk is a dict: {"text": str, "source": "chat|audio|video|text|about", "ref": str}.
Fine for tens of thousands of chunks; swap in LanceDB/Chroma later if you outgrow it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

EmbedFn = Callable[[Sequence[str]], np.ndarray]


class MemoryStore:
    def __init__(self, folder: Path, embed_model: str = "", embed_fn: EmbedFn | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model
        self._embed_fn = embed_fn
        self._model = None
        self.chunks: list[dict] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._load()

    # ---- embedding -------------------------------------------------------
    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._embed_fn is not None:
            vecs = np.asarray(self._embed_fn(texts), dtype=np.float32)
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.embed_model)
            vecs = self._model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=len(texts) > 200
            ).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    # ---- persistence -----------------------------------------------------
    @property
    def _chunks_path(self) -> Path:
        return self.folder / "chunks.json"

    @property
    def _vectors_path(self) -> Path:
        return self.folder / "vectors.npy"

    def _load(self) -> None:
        if self._chunks_path.exists() and self._vectors_path.exists():
            self.chunks = json.loads(self._chunks_path.read_text(encoding="utf-8"))
            self.vectors = np.load(self._vectors_path)

    def save(self) -> None:
        self._chunks_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        np.save(self._vectors_path, self.vectors)

    # ---- API -------------------------------------------------------------
    def add(self, chunks: list[dict], batch_size: int = 64) -> None:
        chunks = [c for c in chunks if c.get("text", "").strip()]
        if not chunks:
            return
        parts = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            parts.append(self._embed([c["text"] for c in batch]))
        new = np.vstack(parts)
        self.vectors = new if self.vectors.size == 0 else np.vstack([self.vectors, new])
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 5, min_score: float = 0.2) -> list[dict]:
        if not self.chunks or not query.strip():
            return []
        q = self._embed([query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in top
            if scores[i] >= min_score
        ]

    def __len__(self) -> int:
        return len(self.chunks)
