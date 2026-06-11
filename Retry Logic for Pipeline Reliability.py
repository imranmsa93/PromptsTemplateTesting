import os
import time

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------- Load Model ----------

def build_model() -> ChatGoogleGenerativeAI:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )


# ---------- Build Analysis Pipeline ----------

def build_pipeline(model: ChatGoogleGenerativeAI):
    """
    Pipeline: user_prompt -> LLM analysis -> text

    This is doing *real* analysis of the user's input, not dummy behavior.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior software architect helping users clarify their ideas. "
                "Given a user input, you must:\n"
                "1) Rewrite it as a clear, detailed requirement.\n"
                "2) Provide at least 3 bullet points describing key considerations.\n"
                "Be explicit and reasonably detailed."
            ),
            (
                "human",
                "User input:\n\n{user_prompt}\n\n"
                "Respond in this format:\n"
                "Rewritten requirement:\n"
                "<detailed paragraph>\n\n"
                "Key considerations:\n"
                "- <point 1>\n"
                "- <point 2>\n"
                "- <point 3 or more>\n"
            ),
        ]
    )

    chain = prompt | model | StrOutputParser()
    return chain


# ---------- Output Quality Validation ----------

def is_output_valid(output_text: str, original_prompt: str) -> bool:
    """
    Simple quality checks to decide if the analysis is 'good enough'.
    This is where the 'actual analysis' aspect comes in.

    Checks:
      - Output length should be at least 1.5x the input length (more detail).
      - Output should contain at least 3 bullet points ('-' at line start).
      - Output must contain the header 'Rewritten requirement:'.
    """
    # Length check
    if len(output_text) < 1.5 * len(original_prompt):
        print(" Output is too short compared to input.")
        return False

    # Bullet point count
    lines = [line.strip() for line in output_text.splitlines()]
    bullet_lines = [l for l in lines if l.startswith("- ")]
    if len(bullet_lines) < 3:
        print(" Not enough bullet points (need at least 3).")
        return False

    # Header check
    if "Rewritten requirement:" not in output_text:
        print(" Missing 'Rewritten requirement:' header.")
        return False

    print(" Output passed validation.")
    return True


# ---------- Retry Helper ----------

def run_with_retry(chain, user_prompt: str, max_attempts: int = 3, delay_seconds: float = 1.5):
    """
    Run the LLM analysis with retry logic.

    We treat 'bad quality output' as a failure and trigger retries.
    This is different from random failures: the user input and model output
    actually influence whether we retry.
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_attempts} ===")

        try:
            # Step 1: Call the LLM pipeline
            output = chain.invoke({"user_prompt": user_prompt})
            print("\n[LLM Raw Output]")
            print(output)

            # Step 2: Validate quality
            print("\n[Validator] Checking output quality...")
            if not is_output_valid(output, user_prompt):
                # Treat validation failure like an exception, so retry logic kicks in
                raise ValueError("Output did not pass quality checks")

            # If we reach here, it's both successful and valid
            print(f"[Retry]  Success on attempt {attempt}")
            return output

        except Exception as e:
            last_error = e
            print(f"[Retry]  Attempt {attempt} failed: {e}")

            if attempt < max_attempts:
                print(f"[Retry]  Retrying in {delay_seconds} seconds...\n")
                time.sleep(delay_seconds)

    print("\n[Retry] All attempts failed. Raising final error.")
    raise last_error


# ---------- MAIN ----------

def main():
    print("\n=== Applying Retry Logic for Pipeline Reliability  ===")

    #  Ask for user input (this is what actually gets analyzed)
    user_prompt = input(
        "\nEnter a requirement / idea you want the assistant to analyze and clarify:\n> "
    )

    model = build_model()
    chain = build_pipeline(model)

    try:
        final_output = run_with_retry(chain, user_prompt)
    except Exception as e:
        print("\n Pipeline failed after all retry attempts:", e)
        return

    print("\n=== Final Accepted Analysis ===")
    print(final_output)


if __name__ == "__main__":
    main()
