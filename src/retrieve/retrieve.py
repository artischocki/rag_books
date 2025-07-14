import faiss
from src.retrieve.index import FaissIndexer
from src.encode.embed import EncoderModel


class Retriever:
    def __init__(
        self,
        encoder: EncoderModel,
        index: FaissIndexer,
        paragraphs: list[str],
    ) -> None:
        self._encoder = encoder
        self._index = index
        self._paragraphs = paragraphs

    def retrieve(self, query: str, k: int = 5):
        # a) bi-encode + FAISS
        q_emb = self._encoder.encode([query])
        faiss.normalize_L2(q_emb)
        D, I = self._index.search(q_emb, k)
        candidates = [self._paragraphs[i] for i in I[0]]

        # b) cross-encode (no embeddings)
        pairs = [[query, doc] for doc in candidates]
        scores = self._encoder.predict(pairs)

        # c) rerank & return top_final
        top = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
        return [c for c, _ in top]
