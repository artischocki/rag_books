import faiss
import numpy as np
from pathlib import Path


def index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product (cosine) similarity
    faiss.normalize_L2(embeddings)  # necessary for cosine
    index.add(embeddings)

    # Optionally: persist index
    print(f"writing index to: {Path(__file__).parents[2] / "book_index.faiss"}")
    faiss.write_index(index, str(Path(__file__).parents[2] / "book_index.faiss"))
    return index
