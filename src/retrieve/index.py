import faiss
import numpy as np
from pathlib import Path


class FaissIndexer:
    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        load_from: Path | None = None,
    ) -> None:
        if load_from is None and embeddings is None:
            return
        if not embeddings is None:
            self._index_embeddings(embeddings)
            return
        if not load_from is None:
            self.load_index(load_from)
            return

        self._index = None
        self._embeddings = None

    def _index_embeddings(self, embeddings) -> None:
        self._embeddings = embeddings
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # inner-product (cosine) similarity
        faiss.normalize_L2(self._embeddings)  # necessary for cosine
        self._index.add(self._embeddings)

    def load_index(self, path: Path) -> None:
        self._index = faiss.read_index(str(path))

    def write_index(self, index_path: Path) -> None:
        if self._index is None:
            raise ValueError("Index has not been created yet.")
        print(f"Writing index to: {index_path}")
        faiss.write_index(self._index, str(index_path))
        return

    def search(self, q_emb: np.ndarray, k: int):
        if self._index is None:
            raise ValueError("Index has not been created yet.")
        return self._index.search(q_emb, k)  # TODO

    @property
    def index(self):
        if self._index is None:
            raise ValueError("Index has not been created yet.")
        return self._index

    @property
    def embeddings(self):
        if self._embeddings is None:
            raise ValueError("No Embeddings saved yet.")
        return self._index
