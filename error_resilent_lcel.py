"""
Implementing Error-Resilient LCEL Pipelines

Scenarios:
- Example 1: Retry-capable LLM pipeline for troubleshooting answers
- Example 2: Validated incident summary pipeline with auto-repair
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Helper: normalize prompt values to plain strings
def _normalize_prompt_value(value) -> str:
    if isinstance(value, tuple) and len(value) == 2:
        # e.g., ("text", "<prompt text>")
        return str(value[1])
    return str(value)

NORMALIZE_PROMPT = RunnableLambda(_normalize_prompt_value)

# LLM bootstrap: Gemini with retry logic, wrapped as RunnableLambda
def build_resilient_llm_runnable(max_retries: int = 3) -> RunnableLambda:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    def call_with_retry(prompt: str) -> str:
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = model.generate_content(prompt)
                if hasattr(response, "text"):
                    return response.text
                # Fallback: join parts if shape is different
                if hasattr(response, "candidates"):
                    texts = []
                    for cand in response.candidates:
                        content = getattr(cand, "content", None)
                        if content and getattr(content, "parts", None):
                            for part in content.parts:
                                if hasattr(part, "text"):
                                    texts.append(part.text)
                    if texts:
                        return "\n".join(texts)
                # If we got here, treat as unexpected format
                return str(response)
            except Exception as e:
                last_exc = e
                print(f"[Retry] Attempt {attempt} failed: {e}")

        return f"[ERROR] Model failed after {max_retries} attempts. Last error: {last_exc}"

    return RunnableLambda(call_with_retry)


# Example 1: Retry-capable troubleshooting pipeline
def build_troubleshooting_chain(llm: RunnableLambda):
    prompt = PromptTemplate.from_template(
        (
            "You are an SRE assistant.\n"
            "Answer the following troubleshooting or operations question clearly and concisely.\n\n"
            "Question:\n{question}\n"
        )
    )

    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | NORMALIZE_PROMPT
        | llm  # retry-enabled LLM
    )

    return chain

# Example 2: Validated incident summary pipeline with auto-repair
def build_incident_summary_chain(llm: RunnableLambda):

    incident_prompt = PromptTemplate.from_template(
    (
        "You are summarizing an incident for an on-call handoff.\n\n"
        "Incident details:\n{incident}\n\n"
        "Write a brief note with exactly two sections:\n"
        "Summary:\n- ...\n\n"
        "Next Actions:\n- ...\n\n"
        "Keep the content concise and focused on what the next on-call needs to know."
    )
    )

    initial_chain = (
        {"incident": RunnablePassthrough()}
        | incident_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    def validate_and_repair(output: str) -> str:
        has_summary = "Summary:" in output
        has_next_actions = "Next Actions:" in output

        if has_summary and has_next_actions:
            return output

        # Auto-repair: enforce the structure
        repair_prompt = (
            "Rewrite the following incident note so that it has exactly two sections:\n"
            "Summary:\n- bullet points\n\n"
            "Next Actions:\n- bullet points\n\n"
            "Keep it short and actionable.\n\n"
            f"Original note:\n{output}"
        )

        # We call the same resilient LLM runnable directly with the repair prompt
        repaired = llm.invoke(repair_prompt)
        return repaired

    validator = RunnableLambda(validate_and_repair)

    # Full pipeline: generate summary, then validate/repair
    full_chain = initial_chain | validator
    return full_chain

#main file 
# CLI entrypoint
def main():
    llm = build_resilient_llm_runnable(max_retries=3)

    print("=" * 60)
    print("Implementing Error-Resilient LCEL Pipelines")
    print("=" * 60)

    # Example 1: troubleshooting with retries
    print("\n[Example 1] Retry-capable troubleshooting pipeline")
    question = input("Enter a troubleshooting / operations question").strip()

    if question:
        troubleshooting_chain = build_troubleshooting_chain(llm)
        answer = troubleshooting_chain.invoke(question)
        print("\n[Answer]\n")
        print(answer)

    # Example 2: validated incident summary
    print("\n[Example 2] Incident summary with validation + auto-repair")
    incident_text = input("Paste a short incident description or log summary ").strip()

    if incident_text:
        incident_chain = build_incident_summary_chain(llm)
        summary = incident_chain.invoke(incident_text)
        print("\n[Incident Handoff Note]\n")
        print(summary)




if __name__ == "__main__":
    main()