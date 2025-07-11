import openai
import os

from src.retrieve.retrieve import retrieve

openai.api_key = os.getenv("OPENAI_API_KEY")


def answer(question: str):
    docs = retrieve(question)
    context = "\n\n---\n\n".join(docs)
    prompt = (
        "You are a helpful assistant. Use the following excerpts from a book to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()
