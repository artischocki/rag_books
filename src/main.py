from src.preprocess.load import load_organized_book
from pathlib import Path


org_book = load_organized_book(
    Path("/home/artur/code/translate-tts/test/translate/org_book.json")
)

print(org_book)
