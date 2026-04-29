import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    # Initialize Gemini model through LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )

    print("=" * 60)
    print("LCEL Chain Examples")
    print("=" * 60)

    # Example 1: Simple Chain (Prompt | LLM | Parser)
    print("\nExample 1: Simple chain")
    print("-" * 60)

    topic = input("Enter a topic to learn an interesting fact about: ")

    prompt = ChatPromptTemplate.from_template(
        "Tell me a short interesting fact about {topic}."
    )

    simple_chain = prompt | llm | StrOutputParser()

    result = simple_chain.invoke({"topic": topic})
    print("\nOutput:", result)

    # Example 2: Reusing the Same Chain with New Input
    print("\nExample 2: Reusing the same chain")
    print("-" * 60)

    new_topic = input("Enter another topic: ")

    result2 = simple_chain.invoke({"topic": new_topic})
    print("\nOutput:", result2)

    # Example 3: Chain with Multiple Variables
    print("\nExample 3: Multi-variable prompt")
    print("-" * 60)

    role = input("Enter a role (e.g., friendly teacher, expert developer): ")
    concept = input("Enter a concept to explain: ")

    complex_prompt = ChatPromptTemplate.from_template(
        "You are a {role}. Explain {concept} in simple terms for a beginner."
    )

    multi_var_chain = complex_prompt | llm | StrOutputParser()

    result3 = multi_var_chain.invoke(
        {"role": role, "concept": concept}
    )

    print("\nOutput:", result3)

    print("\nEnd of LCEL chain demo.")


if __name__ == "__main__":
    main()