import os
from dotenv import load_dotenv
from personas import PERSONAS
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ENGAGEMENT STYLE OPTIONS
STYLE_OPTIONS = {
    "1": "Provide a short, concise explanation.",
    "2": "Provide a deep, technical, research-style explanation.",
    "3": "Provide a code-focused explanation using examples.",
    "4": "Provide a real-world analogy or practical example."
}

# GENERATE RESPONSE FUNCTION
def generate_response(persona_key, user_query, style_pref):
    persona = PERSONAS.get(persona_key)
    if not persona:
        return "Persona not found!"

    role = persona["role"]
    tone = persona["tone"]
    style_instruction = STYLE_OPTIONS.get(style_pref, STYLE_OPTIONS["1"])

    prompt = f"""
### Persona Role ###
{role}

### Persona Speaking Style ###
{tone}

### User Selected Response Style ###
{style_instruction}

### User Query ###
{user_query}

### Persona Response ###
"""

    response = model.invoke(prompt)
    return response.content


# MAIN LOOP
def main():
    print("\n=== Tech Persona Context Injection ===\n")
    print("Available Personas:")
    for p in PERSONAS.keys():
        print(f"- {p}")

    current_persona = input("\nChoose a persona: ").strip()

    while True:
        print("\nHow do you want the response?")
        print("1. Short explanation")
        print("2. Deep technical explanation")
        print("3. Code-focused explanation")
        print("4. Real-world example")

        style_pref = input("\nSelect 1-4: ").strip()
        user_query = input("\nEnter your question: ")

        output = generate_response(current_persona, user_query, style_pref)
        print(output)

        # -------- ASK IF USER WANTS TO CHANGE PERSONA --------
        switch = input("Switch persona? (y/n): ").strip().lower()
        if switch == "y":
            print("\nAvailable Personas:")
            for p in PERSONAS.keys():
                print(f"- {p}")
            current_persona = input("\nEnter new persona: ").strip()

        # -------- CONTINUE OR EXIT --------
        cont = input("\nDo you want to continue? (y/n): ").strip().lower()
        if cont != "y":
            break


if __name__ == "__main__":
    main()