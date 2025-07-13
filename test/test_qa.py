import json
import pytest
from pathlib import Path
from openai import OpenAI

from src.rag import BookRag

# load all four suites
with open(Path(__file__).parent / "resources" / "qna_pairs.json") as f:
    all_qna_pairs = json.load(f)
    QA_FACTUAL_RECALL = all_qna_pairs["Factual / Recall"]
    QA_COMPREHENSION = all_qna_pairs["Comprehension"]
    QA_ANALYSIS = all_qna_pairs["Analysis"]
    QA_SYNTHESIS_EVALUATION = all_qna_pairs["Synthesis / Evaluation"]

# flatten into a list of (suite_name, qa) tuples
ALL_QA = (
    [("Factual / Recall", qa) for qa in QA_FACTUAL_RECALL]
    + [("Comprehension", qa) for qa in QA_COMPREHENSION]
    + [("Analysis", qa) for qa in QA_ANALYSIS]
    + [("Synthesis / Evaluation", qa) for qa in QA_SYNTHESIS_EVALUATION]
)

# initialize your RAG system and OpenAI client
org_book_path = Path(__file__).parents[1] / "test" / "resources" / "org_book.json"
BOOK_RAG = BookRag(org_book_path)
CLIENT = OpenAI()


def evaluate_with_llm(student_answer: str, reference: str, client) -> dict:
    """
    Ask the LLM to compare student_answer vs. reference.
    Returns a dict with keys: "correct" (bool) and "explanation" (str).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that evaluates whether a student's answer "
                "correctly matches the reference answer. "
                "Respond STRICTLY in JSON with two keys: "
                "'correct' (true or false) and 'explanation' (a brief justification)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Reference Answer:\n{reference}\n\n"
                f"Student's Answer:\n{student_answer}\n\n"
                "Is the student's answer correct? "
                "Reply only with the JSON."
            ),
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4", messages=messages, temperature=0.0
    )
    return json.loads(resp.choices[0].message.content.strip())


@pytest.mark.parametrize("suite_name, qa", ALL_QA)
def test_question(request, suite_name, qa):
    generated = BOOK_RAG.answer(qa["question"])
    result = evaluate_with_llm(generated, qa["answer"], CLIENT)

    # record for the final report in conftest.py
    request.node.user_properties.append(("question", qa["question"]))
    request.node.user_properties.append(("reference", qa["answer"]))
    request.node.user_properties.append(("generated", generated))
    request.node.user_properties.append(("correct", result["correct"]))
    request.node.user_properties.append(("explanation", result["explanation"]))

    assert result["correct"], (
        f"[{suite_name}] Q: {qa['question']}\n"
        f"Ref: {qa['answer']}\n"
        f"Got: {generated}\n"
        f"Explain: {result['explanation']}"
    )
