from src.rag import BookRag

from pathlib import Path

org_book_path = Path(__file__).parents[1] / "test" / "resources" / "org_book.json"

book_rag = BookRag(org_book_path)

# print(answer("Which characters are there?"))
# print(generator.answer("Whose father is Nicolai Lvovitch?"))
# print(generator.answer("Who is Mussolini?"))
while True:
    question = input("Question: ")
    print(f"Answer:   {book_rag.answer(question)}\n")
