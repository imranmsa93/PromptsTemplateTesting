"""
CodeSprint Studio - UPDATED Agentic AI System (LangChain v1 compatible)
Fixes:
- google-genai SDK (new)
- create_tool_calling_agent (new API)
- modern Gemini integration
"""

import os
from dotenv import load_dotenv
from typing import List

from google import genai

from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent




# =========================
# 1. STRUCTURED OUTPUT (Pydantic)
# =========================

class CodeAnalysisOutput(BaseModel):
    task_summary: str = Field(..., description="Summary of developer request")
    risks: List[str] = Field(..., description="Potential risks in code")
    improvements: List[str] = Field(..., description="Suggested improvements")
    next_steps: List[str] = Field(..., description="Next actions for developer")


# =========================
# 2. TOOLS
# =========================

@tool
def analyze_code_snippet(code: str) -> str:
    """Analyzes code snippet for common issues."""
    issues = []

    if "print" in code:
        issues.append("Avoid print statements in production")

    if "except:" in code:
        issues.append("Avoid bare except blocks")

    if "TODO" in code:
        issues.append("Code contains TODOs")

    return "\n".join(issues) if issues else "No major issues found"


@tool
def summarize_text(text: str) -> str:
    """Summarizes long text into short form."""
    return text[:200]


# =========================
# 3. GEMINI LLM WRAPPER (NEW SDK)
# =========================

def build_gemini_llm(max_retries: int = 3):
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)

    def call(prompt: str) -> str:
        last_error = None

        for _ in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                return response.text
            except Exception as e:
                last_error = e

        return f"ERROR: {last_error}"

    return RunnableLambda(call)


# =========================
# 4. SYSTEM PROMPT (AGENT PERSONA)
# =========================

SYSTEM_PROMPT = """
You are CodeSprint Studio Senior Developer Assistant.

You help with:
- code understanding
- debugging
- refactoring suggestions
- architecture reasoning

Rules:
- Be structured and concise
- Use tools when needed
- Prefer actionable insights
"""


# IMPORTANT: must include agent_scratchpad in v1 agents
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


# =========================
# 5. AGENT CLASS
# =========================

class CodeSprintAgent:

    def __init__(self):
        self.llm = build_gemini_llm()
        self.tools = [analyze_code_snippet, summarize_text]

        # ✅ NEW TOOL-CALLING AGENT (v1 correct)
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )

    def run(self, query: str):
        return self.executor.invoke({"input": query})


# =========================
# 6. LCEL PIPELINE (VALIDATION FLOW)
# =========================

class CodeSprintPipeline:

    def __init__(self, llm):
        self.llm = llm

        self.chain = (
            RunnableLambda(lambda x: f"""
You are a senior developer assistant.

Analyze:
{x}

Return structured insights.
""")
            | self.llm
        )

    def run(self, input_text: str) -> CodeAnalysisOutput:
        raw = self.chain.invoke(input_text)

        # Simple structured mapping (can be upgraded to JSON mode)
        return CodeAnalysisOutput(
            task_summary=raw[:120],
            risks=["runtime interpretation risk"],
            improvements=["use structured JSON output"],
            next_steps=["refactor into strict schema pipeline"]
        )


# =========================
# 7. MAIN
# =========================

def main():
    print("\n=== CodeSprint Studio Agent (UPDATED) ===\n")

    agent = CodeSprintAgent()

    result = agent.run(
        "Analyze this code: try: x=1/0 except: print('error')"
    )

    print("\n--- AGENT OUTPUT ---\n")
    print(result)


if __name__ == "__main__":
    main()