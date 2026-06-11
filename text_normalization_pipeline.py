import re
from typing import Callable, List

PipelineStep = Callable[[str], str]


# ---------- Normalization Steps ----------

def strip_whitespace(text: str) -> str:
    return text.strip()

def to_lowercase(text: str) -> str:
    return text.lower()

def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)

def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)

def normalize_numbers(text: str) -> str:
    return re.sub(r"\d", "#", text)


# ---------- Pipeline Runner ----------

def run_pipeline(text: str, steps: List[PipelineStep]) -> str:
    for step in steps:
        text = step(text)
    return text


# ---------- Runner ----------

def run_text_normalization_demo():
    print("\n=== Text Normalization Pipeline Demo (Interactive) ===\n")

    # Ask the user for input
    user_input = input("Enter any text to normalize:\n> ")

    pipeline_steps = [
        strip_whitespace,
        to_lowercase,
        collapse_spaces,
        remove_punctuation,
        normalize_numbers,
    ]

    print("\n--- Processing Your Input ---")
    print("Raw Text:       ", repr(user_input))

    normalized = run_pipeline(user_input, pipeline_steps)

    print("Normalized Text:", repr(normalized))
    print("\n=== Demo Complete ===\n")


if __name__ == "__main__":
    run_text_normalization_demo()
