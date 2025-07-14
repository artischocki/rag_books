from sentence_transformers import SentenceTransformer  # This takes a long ass time
from sentence_transformers import CrossEncoder
import numpy as np


class EncoderModel:
    def __init__(self, model: str | None = None):
        if model is None:
            model = "all-MiniLM-L6-v2"  # small & fast

        self._embed_model = SentenceTransformer(model)

    def _encode_paragraphs(self, paragraphs: list[str]):
        # Compute embeddings in batches
        embeddings = self._embed_model.encode(
            paragraphs, batch_size=32, show_progress_bar=True
        )
        return embeddings

    def encode(self, query: list[str]) -> np.ndarray:
        return self._embed_model.encode(query)


class Encoder:
    def __init__(self):
        self._bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
