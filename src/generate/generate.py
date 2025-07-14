from openai import OpenAI
import os

from src.encode.embed import EncoderModel
from src.retrieve.retrieve import Retriever
from src.retrieve.index import FaissIndexer


class OpenAiGenerator:
    def __init__(
        self, encoder: EncoderModel, indexer: FaissIndexer, paragraphs: list[str]
    ):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._encoder = encoder
        self._indexer = indexer
        self._paragraphs = paragraphs
        self._retriever = Retriever(self._encoder, self._indexer, self._paragraphs)

    def answer(
        self,
        question: str,
    ):
        docs = self._retriever.retrieve(
            question,
        )
        context = "\n\n---\n\n".join(docs)
        prompt = f"""
            You are an assistant that answers questions related to a book.

            The excerpts below were retrieved from the book.
            They will help you answer the question.

            1. Focus strictly on the provided excerpts:
               - Do not introduce facts or details that are not present in the excerpts.
               - If the answer isn’t contained in the excerpts, say:  
                 'No Information.'
            2. Answer style:
               - Be helpful, precise, and to the point.
               - It is not necessary to reference the provided text in your answers.
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
