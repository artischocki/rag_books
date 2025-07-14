from src.preprocess.load import load_organized_book
from src.encode.embed import EncoderModel
from src.retrieve.index import FaissIndexer
from src.generate.generate import OpenAiGenerator

from pathlib import Path


class BookRag:
    def __init__(self, org_book_path: Path) -> None:
        org_book = load_organized_book(org_book_path)

        self._encoder_model = EncoderModel()
        self._faiss_index = FaissIndexer()
        self._paragraphs = []

        # flatten book: TODO i'm not sure if this will be necessary in the future
        for part in org_book.values():
            for chapter in part.values():
                self._paragraphs += chapter

        book_index_path = Path(__file__).parents[1] / "book_index.faiss"

        if book_index_path.exists():
            self._faiss_index.load_index(book_index_path)
        else:
            embeddings = self._encoder_model._embed_paragraphs(self._paragraphs)
            self._faiss_index._index_embeddings(embeddings)

        self._generator = OpenAiGenerator(
            self._encoder_model, self._faiss_index, self._paragraphs
        )

    def answer(self, question: str) -> str:
        return self._generator.answer(question)
