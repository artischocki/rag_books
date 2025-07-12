from openai import OpenAI
import os

from src.preprocess.embed import EmbeddingModel
from src.retrieve.retrieve import retrieve


class OpenAIGenerator:
    def __init__(self, embedding_model: EmbeddingModel, index, paragraphs):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._embedding_model = embedding_model
        self._index = index
        self._paragraphs = paragraphs

    def answer(
        self,
        question: str,
    ):
        docs = retrieve(
            self._embedding_model._embed_model, self._index, self._paragraphs, question
        )
        context = "\n\n---\n\n".join(docs)
        prompt = (
            "You are a helpful assistant. Use the following excerpts from a book to answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        resp = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
