import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence


# ---------- Data model ----------

@dataclass
class IdeaScore:
    idea: str
    score: float
    reason: str


# ---------- Helpers ----------

def load_model() -> ChatGoogleGenerativeAI:
    """Load API key from .env and create the LLM client."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set in .env. "
            "Please create a .env file with your Google API key."
        )

    # Use a lightweight model; adjust as needed
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=api_key,
        temperature=0.2,
    )


def build_scoring_chain(model: ChatGoogleGenerativeAI) -> RunnableSequence:
    """
    Build a small LangChain pipeline:
    PromptTemplate -> Model
    We'll parse the score from the model's text output.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful product manager assistant. "
                    "Given a feature idea, assign a usefulness score from 1.0 to 5.0. "
                    "Higher means more impact for end users.\n"
                    "Return your answer in this format:\n"
                    "Score: <number>\n"
                    "Reason: <short explanation>"
                ),
            ),
            (
                "human",
                "Feature idea: {idea}\n\nProvide the score and reason.",
            ),
        ]
    )

    chain = prompt | model
    return chain


def parse_score_and_reason(text: str) -> IdeaScore:
    """
    Extract numeric score and reason text from the LLM output.
    Assumes format:
        Score: 4.5
        Reason: Some explanation...
    """

    # Find a number like 1, 4.3, 5.0, etc.
    score_match = re.search(r"Score\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
    else:
        # Fallback if parsing fails
        score = 0.0

    # Get the reason line or everything after "Reason:"
    reason_match = re.search(r"Reason\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        reason = "No reason provided."

    # We'll fill 'idea' later; this function only parses raw text
    return score, reason


# ---------- Demo runner ----------

def run_scoring_and_ranking_demo() -> None:
    print("=== Scoring and Ranking Processor Demo ===\n")

    # 1) Sample items to score
    feature_ideas = [
        "Add a dark mode theme to the dashboard.",
        "Send a handwritten postcard to every new user.",
        "Implement one-click export of reports to Excel and PDF.",
        "Show a daily motivational quote on the home page.",
        "Add real-time collaboration so multiple users can edit a document together.",
    ]

    print("Feature ideas to score:")
    for idea in feature_ideas:
        print(f"- {idea}")
    print("\nScoring ideas with the LLM...\n")

    # 2) Build the model and scoring chain
    model = load_model()
    chain = build_scoring_chain(model)

    scored_ideas: List[IdeaScore] = []

    for idea in feature_ideas:
        # Invoke the chain and get the raw LLM output text
        response = chain.invoke({"idea": idea})
        # langchain-google-genai returns an object with .content
        raw_text = response.content if hasattr(response, "content") else str(response)

        score, reason = parse_score_and_reason(raw_text)
        scored_ideas.append(IdeaScore(idea=idea, score=score, reason=reason))

        print(f"Idea:   {idea}")
        print(f"Score:  {score}")
        print(f"Reason: {reason}\n")

    # 3) Rank by score, highest first
    scored_ideas.sort(key=lambda i: i.score, reverse=True)

    print("\n=== Ranked Ideas (Highest Score First) ===\n")
    for rank, item in enumerate(scored_ideas, start=1):
        print(f"{rank}. [{item.score:.1f}] {item.idea}")
        print(f"   Reason: {item.reason}\n")


if __name__ == "__main__":
    run_scoring_and_ranking_demo()
