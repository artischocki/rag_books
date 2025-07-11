from src.preprocess.load import load_organized_book
from src.preprocess.embed import EmbeddingModel
from src.preprocess.store import index
from src.retrieve.retrieve import retrieve

from pathlib import Path


org_book = load_organized_book(
    Path("/home/artur/code/translate-tts/test/translate/org_book.json")
)

embed_model = EmbeddingModel()
# paragraphs = org_book["PART I"]["I."]
paragraphs = []

# flatten book:
# im just testing here
for part in org_book.values():
    for chapter in part.values():
        paragraphs += chapter

embeddings = embed_model._embed_paragraphs(paragraphs)
index = index(embeddings)


# QUICK POC
from openai import OpenAI
import os

from src.retrieve.retrieve import retrieve

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def answer(question: str):
    docs = retrieve(embed_model._embed_model, index, paragraphs, question)
    context = "\n\n---\n\n".join(docs)
    prompt = (
        "You are a helpful assistant. Use the following excerpts from a book to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# print(answer("Which characters are there?"))
print(answer("Whose father is Nicolai Lvovitch?"))
print(answer("Who is Mussolini?"))
