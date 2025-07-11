import faiss
import numpy as np


def index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product (cosine) similarity
    faiss.normalize_L2(embeddings)  # necessary for cosine
    index.add(embeddings)

    # Optionally: persist index
    faiss.write_index(index, "../../book_index.faiss")
    return index
