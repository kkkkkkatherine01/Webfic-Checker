"""Text embeddings for passage search, computed locally (no API cost).

The real embedder runs BAAI/bge-small-zh-v1.5 through fastembed (ONNX, no PyTorch). It
is loaded on first use, so commands that never search do not pay for it.
"""

from pathlib import Path
from typing import Protocol

# bge models expect this instruction before search queries, not before passages.
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model_name: str, cache_dir: Path | None = None, dim: int = 512):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.dim = dim
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # heavy import, only when needed

            cache = str(self.cache_dir / "fastembed") if self.cache_dir else None
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=cache)
        return self._model

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self._load().embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().embed([BGE_QUERY_INSTRUCTION + text]))).tolist()
