import json
import os
import pytest
from pathlib import Path
from openai import OpenAI
from .teacher_llm import evaluate_with_llm

from src.rag import BookRag


with open(Path(__file__).parent / "resources" / "qna_pairs.json") as f:
    all_qna_pairs = json.load(f)


SUITES = [
    ("Factual / Recall", all_qna_pairs["Factual / Recall"]),
    ("Comprehension", all_qna_pairs["Comprehension"]),
    ("Analysis", all_qna_pairs["Analysis"]),
    ("Synthesis / Evaluation", all_qna_pairs["Synthesis / Evaluation"]),
]

org_book_path = Path(__file__).parents[1] / "test" / "resources" / "org_book.json"
BOOK_RAG = BookRag(org_book_path)

CLIENT = OpenAI()

OUTPUT_JSON = Path(__file__).parent / "eval_results" / "test_result_no_adaptations.json"
if OUTPUT_JSON.exists():
    raise FileExistsError(
        f"The file {OUTPUT_JSON} already exists. Please change the name of the output json, so the results can later be evaluated."
    )


# flatten into (suite_name, question, answer)
cases = []
for suite_name, qa_list in SUITES:
    for qa in qa_list:
        cases.append((suite_name, qa["question"], qa["answer"]))


@pytest.mark.parametrize("suite_name, question, reference", cases)
def test_single_qa(request, suite_name, question, reference):
    generated = BOOK_RAG.answer(question)
    result = evaluate_with_llm(generated, reference, CLIENT)

    # load output json or init
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # ensure list for this suite
    suite_list = all_results.setdefault(suite_name, [])
    correct = result["correct"]
    suite_list.append(
        {
            "question": question,
            "reference": reference,
            "generated": generated,
            "correct": correct,
            "explanation": result["explanation"],
        }
    )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=4)

    # assert after writing
    assert result["correct"], (
        f"[{suite_name}] Q: {question}\n"
        f"Ref: {reference}\nGot: {generated}\n"
        f"Explain: {result['explanation']}"
    )
