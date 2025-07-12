from src.preprocess.load import load_organized_book
from src.preprocess.embed import EmbeddingModel
from src.preprocess.index import FaissIndex
from src.generate.generate import OpenAIGenerator

from pathlib import Path


org_book = load_organized_book(Path(__file__).parents[1] / "test" / "org_book.json")

embed_model = EmbeddingModel()
faiss_index = FaissIndex()
paragraphs = []

# flatten book:
# im just testing here
for part in org_book.values():
    for chapter in part.values():
        paragraphs += chapter

book_index_path = Path(__file__).parents[1] / "book_index.faiss"

if book_index_path.exists():
    faiss_index.load_index(book_index_path)
else:
    embeddings = embed_model._embed_paragraphs(paragraphs)
    faiss_index._index_embeddings(embeddings)


# QUICK POC

generator = OpenAIGenerator(embed_model, faiss_index.index, paragraphs)


# print(answer("Which characters are there?"))
# print(generator.answer("Whose father is Nicolai Lvovitch?"))
# print(generator.answer("Who is Mussolini?"))
while True:
    question = input("Question: ")
    print(f"Answer:   {generator.answer(question)}\n")
