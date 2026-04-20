import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

# Method 1. Zero-Shot Prompting
def zero_shot_prompt():
    print("\n--- Zero-Shot Prompt ---\n")
    topic = input("Enter a topic: ")

    template = "Explain the topic: {topic} in simple language."
    final_prompt = PromptTemplate(
        input_variables=["topic"],
        template=template
    ).format(topic=topic)

    print("\nGenerated Prompt:\n", final_prompt)
    response = llm.invoke(final_prompt)
    print("\nGemini Response:\n", response.content)


# Method 2. Few-Shot Prompting
def few_shot_prompt():
    print("\n--- Few-Shot Prompt ---\n")
    topic = input("Enter a topic: ")

    template = """
Provide a clear explanation for the topic below, following the same tone, structure, and depth as standard scientific explanations.

Topic: {topic}
Explanation:
"""

    final_prompt = PromptTemplate(
        input_variables=["topic"],
        template=template
    ).format(topic=topic)

    print("\nGenerated Prompt:\n", final_prompt)
    response = llm.invoke(final_prompt)
    print("\nGemini Response:\n", response.content)


# Method 3. Iterative / Chain-of-Thought Prompting
def iterative_prompt():
    print("\n--- Iterative / Chain-of-Thought Prompt ---\n")
    topic = input("Enter a topic: ")

    # Step 1: Generate a brief explanation
    step1_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} briefly."
    ).format(topic=topic)

    step1_resp = llm.invoke(step1_prompt)
    print("\nStep 1 Output:\n", step1_resp.content)

    # Step 2: Expand using examples or analogies
    step2_prompt = PromptTemplate(
        input_variables=["prev", "topic"],
        template="Using the explanation below:\n{prev}\nProvide 2 examples or analogies for {topic}."
    ).format(prev=step1_resp.content, topic=topic)

    step2_resp = llm.invoke(step2_prompt)

    print("\nStep 2 Output (Examples):\n", step2_resp.content)

    # Method 4. High-Quality Prompt Construction
def high_quality_prompt():
    print("\n--- High-Quality Prompt ---\n")
    topic = input("Enter a topic: ")
    audience = input("Target audience (beginner / student / professional): ")
    tone = input("Preferred tone (friendly / formal / expert): ")

    template = """
You are an expert instructor.
Audience: {audience}
Tone: {tone}
Explain the topic clearly and concisely.
Include examples or analogies when appropriate.
Topic: {topic}
Answer:
"""

    final_prompt = PromptTemplate(
        input_variables=["topic", "audience", "tone"],
        template=template
    ).format(topic=topic, audience=audience, tone=tone)

    print("\nGenerated Prompt:\n", final_prompt)
    response = llm.invoke(final_prompt)
    print("\nGemini Response:\n", response.content)


# Method 5. Chat-Oriented Prompting
def chat_prompt():
    print("\n--- Chat Prompt ---\n")
    system_role = input("System role: ")
    user_message = input("User message: ")
    tone = input("AI tone (friendly / strict / expert): ")

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "{system_role}"),
        ("user", "{user_message}"),
        ("assistant", "Respond in a {tone} tone.")
    ])

    formatted_messages = chat_template.format_messages(
        system_role=system_role,
        user_message=user_message,
        tone=tone
    )

    print("\nGenerated Chat Prompt:")
    for msg in formatted_messages:
        print(f"{msg.type.upper()}: {msg.content}")

    print("\nGenerating response...\n")
    response = llm.invoke(formatted_messages)
    print("Gemini Reply:\n", response.content)


# Main Menu
def main():
    print("\n=== Gemini Prompt Engineering (LangChain) ===\n")
    print("Select a technique:")
    print("1. Zero-Shot Prompt")
    print("2. Few-Shot Prompt")
    print("3. Iterative / Chain-of-Thought Prompt")
    print("4. High-Quality Prompt")
    print("5. Chat Prompt")
    print("Type exit() to quit.\n")

    while True:
        choice = input("Choose 1-5: ").strip()

        if choice.lower() in ["exit", "exit()", "quit"]:
            print("\nExiting...\n")
            break

        if choice == "1":
            zero_shot_prompt()
        elif choice == "2":
            few_shot_prompt()
        elif choice == "3":
            iterative_prompt()
        elif choice == "4":
            high_quality_prompt()
        elif choice == "5":
            chat_prompt()
        else:
            print("Invalid selection. Please choose from 1-5.\n")


if __name__ == "__main__":
    main()