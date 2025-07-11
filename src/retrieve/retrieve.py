import faiss


def retrieve(embed_model, index, paragraphs, query: str, k: int = 5):
    q_emb = embed_model.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [paragraphs[i] for i in I[0]]
