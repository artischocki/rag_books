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
        prompt = f"""
            You are an assistant that answers questions related to a book.

            The excerpts below were retrieved from the book.
            They will help you answer the question.

            1. Focus strictly on the provided excerpts:
               - Do not introduce facts or details that are not present in the excerpts.
               - If the answer isn’t contained in the excerpts, say:  
                 'The provided text does not contain that information.'
            2. Answer style:
               - Be helpful, precise, and to the point.
               - For interpretive or analytical questions, make clear that your interpretation is grounded in the text.
               - If the user asks for a summary, synthesize only what’s in the retrieved excerpts.
            
            Excerpts:
            {context}

            Question:
            {question}
            """
        resp = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip(), docs
