from pathlib import Path
import json


def load_organized_book(path: Path) -> dict[str, dict[str, list[str]]]:
    """
    You have to preprocess your raw txt book into a json that follows this
    structure:
        "Part 1":
            "Chapter 1":
                listof[paragraphs]
            "Chapter 2":
                listof[paragraphs]
            ...
        "Part 2":
            "Chapter 1":
                listof[paragraphs]
            "Chapter 2":
                listof[paragraphs]
            ...
        ...

    Args:
        path: has to be path to a json
    """
    with open(path) as f:
        org_book = json.load(f)

    return org_book
