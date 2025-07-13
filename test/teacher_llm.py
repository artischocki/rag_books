import json


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
