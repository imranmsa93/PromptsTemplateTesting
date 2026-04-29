"""
Designing Multi-Step LCEL Workflows

Scenarios:
- Example 1: Feature design workflow (idea → requirements → technical spec)
- Example 2: Service analysis workflow (summary + risks + tests in parallel)
"""


import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableMap,
)


# LLM bootstrap: Gemini via google-generativeai, wrapped for LCEL
def build_llm_runnable() -> RunnableLambda:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    def call_gemini(prompt: str) -> str:
        """Call Gemini with a text prompt and return text output."""
        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            return response.text

        # Fallback in case response format changes
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

        return str(response)

    return RunnableLambda(call_gemini)


# Helper: normalize prompt values to plain strings
def _normalize_prompt_value(value) -> str:
    if isinstance(value, tuple) and len(value) == 2:
        return str(value[1])
    return str(value)


NORMALIZE_PROMPT = RunnableLambda(_normalize_prompt_value)


# Example 1: Feature design workflow (sequential multi-step)
def build_feature_design_chain(llm: RunnableLambda):
    """
    Step 1: Turn high-level idea into structured requirements.
    Step 2: Turn requirements into a technical spec.
    """

    # Step 1: idea -> requirements
    requirements_prompt = PromptTemplate.from_template(
        (
            "You are a product-minded engineer.\n"
            "Convert the following feature idea into clear, structured requirements:\n"
            "{feature_idea}\n\n"
            "Organize the output under headings:\n"
            "- User Stories\n"
            "- Functional Requirements\n"
            "- Non-Functional Requirements\n"
        )
    )

    requirements_chain = (
        requirements_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    # Step 2: requirements -> technical spec
    spec_prompt = PromptTemplate.from_template(
        (
            "You are a senior backend engineer.\n"
            "Based on the following requirements, propose a technical design.\n\n"
            "Requirements:\n{requirements}\n\n"
            "Include sections:\n"
            "- High-Level Architecture\n"
            "- Data Model (entities + key fields)\n"
            "- API Design (endpoints or method signatures)\n"
            "- Validation Rules\n"
            "- Important Edge Cases\n"
        )
    )

    multi_step_chain = (
        {"feature_idea": RunnablePassthrough()}
        | {"requirements": requirements_chain}
        | spec_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    return multi_step_chain


# Example 2: Service analysis workflow (parallel branches)
def build_service_analysis_chain(llm: RunnableLambda):
    """
    Parallel branches:
    - summary
    - risks
    - tests
    """

    summary_prompt = PromptTemplate.from_template(
        (
            "You are explaining this service to a new team member:\n"
            "{service}\n\n"
            "Describe:\n"
            "- What the service is responsible for\n"
            "- Typical dependencies\n"
            "- When you would use it in a system\n"
        )
    )

    risks_prompt = PromptTemplate.from_template(
        (
            "Consider the following service:\n"
            "{service}\n\n"
            "List potential risks and failure modes under headings:\n"
            "- Functional Risks\n"
            "- Performance / Scalability Risks\n"
            "- Operational Risks\n"
        )
    )

    tests_prompt = PromptTemplate.from_template(
        (
            "For the service:\n"
            "{service}\n\n"
            "Propose a testing strategy. Include bullet lists for:\n"
            "- Unit Tests\n"
            "- Integration Tests\n"
            "- Load / Performance Tests\n"
        )
    )

    summary_chain = (
        {"service": RunnablePassthrough()}
        | summary_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    risks_chain = (
        {"service": RunnablePassthrough()}
        | risks_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    tests_chain = (
        {"service": RunnablePassthrough()}
        | tests_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    parallel_branches = RunnableMap(
        {
            "summary": summary_chain,
            "risks": risks_chain,
            "tests": tests_chain,
        }
    )

    return parallel_branches


def main():
    llm = build_llm_runnable()

    print("-" * 60)
    print("Demo 2: Designing Multi-Step LCEL Workflows")
    print("-" * 60)

    # Example 1
    print("\n[Example 1] Feature design: idea + requirements + technical spec")
    feature_idea = input("Enter a feature idea: ").strip()

    if feature_idea:
        feature_chain = build_feature_design_chain(llm)
        spec = feature_chain.invoke(feature_idea)

        print("\n[Generated Technical Spec]\n")
        print(spec)

    # Example 2
    print("\n[Example 2] Service analysis: summary + risks + tests (parallel)")
    service_name = input("Enter a service or component name: ").strip()

    if service_name:
        service_chain = build_service_analysis_chain(llm)
        analysis = service_chain.invoke(service_name)

        print("\n[Service Summary]\n")
        print(analysis["summary"])

        print("\n[Risks & Failure Modes]\n")
        print(analysis["risks"])

        print("\n[Test Strategy]\n")
        print(analysis["tests"])


if __name__ == "__main__":
    main()