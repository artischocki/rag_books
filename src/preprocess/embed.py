print("Importing sentence_transformers...")
from sentence_transformers import SentenceTransformer  # This takes a long ass time

print("Done.")


class EmbeddingModel:
    def __init__(self, model: str | None = None):
        if model is None:
            model = "all-MiniLM-L6-v2"  # small & fast

        self._embed_model = SentenceTransformer(model)

    def _embed_paragraphs(self, paragraphs: list[str]):
        # Compute embeddings in batches
        embeddings = self._embed_model.encode(
            paragraphs, batch_size=32, show_progress_bar=True
        )
        return embeddings
