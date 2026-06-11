import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
)

# 1. CONFIGURE GEMINI FROM .env
load_dotenv()
API_KEY = os.getenv("API_Key")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not API_KEY:
    raise ValueError("API_Key not found in .env file")

print(f"Using Gemini model: {MODEL_NAME}")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME, generation_config={"temperature": 0.3})


def call_gemini(prompt: str) -> str:
    """Call Gemini and return plain text."""
    resp = model.generate_content(prompt)
    return resp.text


# 2. INPUT NORMALIZATION: TICKET + USER-CHOSEN URGENCY

def collect_inputs(user_input: dict) -> dict:
    """
    user_input: {"ticket": str, "urgency_choice": str}

    urgency_choice is expected to be "1", "2", or "3".
    We map it to human-readable labels.
    """
    ticket_text = user_input["ticket"]
    choice = user_input["urgency_choice"].strip()

    mapping = {
        "1": "high",
        "2": "medium",
        "3": "low",
    }

    urgency_label = mapping.get(choice, "low")

    result = {
        "ticket": ticket_text,
        "urgency": urgency_label,
    }

    print("Collected inputs:", result)
    return result


collector_runnable = RunnableLambda(collect_inputs)

# 3. PARALLEL STAGE 
parallel_stage = RunnableParallel(
    ticket=RunnableLambda(lambda d: d["ticket"]),
    urgency=RunnableLambda(lambda d: d["urgency"]),
    original=RunnablePassthrough(),
)

# 4. BRANCH HANDLERS

def high_urgency_response(data: dict) -> str:
    urgency = data["urgency"]
    ticket_text = data["ticket"]

    prompt = f"""
You are a senior on-call support engineer.

Urgency: {urgency.upper()}
Ticket:
\"\"\"{ticket_text}\"\"\"

Write a short escalation note for the on-call engineer that:
- Explains the issue
- Emphasizes it's high priority / production-impacting
- Suggests the first troubleshooting step.
"""
    return call_gemini(prompt)


def medium_urgency_response(data: dict) -> str:
    urgency = data["urgency"]
    ticket_text = data["ticket"]

    prompt = f"""
You are a support engineer replying to a customer.

Urgency: {urgency}
Ticket:
\"\"\"{ticket_text}\"\"\"

Write a short, friendly email that:
- Acknowledges the issue
- Explains that the team is investigating
- Provides any immediate workaround if possible.
"""
    return call_gemini(prompt)


def low_urgency_response(data: dict) -> str:
    urgency = data["urgency"]
    ticket_text = data["ticket"]

    prompt = f"""
You are a support agent replying to a low-urgency feature request or minor issue.

Urgency: {urgency}
Ticket:
\"\"\"{ticket_text}\"\"\"

Write a polite response that:
- Thanks the user for their feedback
- Explains it will be handled in the normal backlog/roadmap
- Keeps the tone calm and professional.
"""
    return call_gemini(prompt)


high_branch = RunnableLambda(high_urgency_response)
medium_branch = RunnableLambda(medium_urgency_response)
low_branch = RunnableLambda(low_urgency_response)

# 5. BRANCH CONDITIONS

def is_high(data: dict) -> bool:
    return data["urgency"].lower() == "high"


def is_medium(data: dict) -> bool:
    return data["urgency"].lower() == "medium"


ticket_router = RunnableBranch(
    (is_high, high_branch),
    (is_medium, medium_branch),
    low_branch,  # default (low)
)

# 6. FULL WORKFLOW
# Overall shape:
#   user_input (dict) -> collect_inputs -> parallel_stage -> ticket_router
workflow = collector_runnable | parallel_stage | ticket_router


if __name__ == "__main__":
    print("=== RunnableBranch Routing with User-Selected Urgency ===")

    ticket_text = input("Paste a support ticket description:\n> ")
    print("\nChoose urgency level:")
    print("  1 = HIGH (critical outage / security / revenue impact)")
    print("  2 = MEDIUM (performance issues / partial impact)")
    print("  3 = LOW (feature request / minor inconvenience)")
    urgency_choice = input("Enter 1, 2, or 3: ")

    user_input = {
        "ticket": ticket_text,
        "urgency_choice": urgency_choice,
    }

    result = workflow.invoke(user_input)

    print("\n--- Routed Support Response ---")
    print(result)
