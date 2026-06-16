# Multi-Hop ReAct Workflow with LangChain

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from typing import Optional

# STEP 1: Define Tools using LangChain

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information. Use this to find facts."""
    knowledge_base = {
        "capital of france": "Paris",
        "population of paris": "2.1 million people",
        "paris location": "Paris is in northern France, on the Seine River",
        "france president": "Emmanuel Macron",
        "emmanuel macron birth": "Emmanuel Macron was born in 1977 in Amiens, France",
        "largest city france": "Paris is the largest city in France",
        "eiffel tower height": "The Eiffel Tower is 330 meters tall",
        "eiffel tower location": "The Eiffel Tower is located in Paris",
    }

    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower or query_lower in key:
            return value
    return "No information found for this query."


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Input should be a valid Python expression."""
    try:
        # Restricted eval (no builtins) for safety in this demo
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"


# STEP 2: Create ReAct Prompt Template

REACT_PROMPT_TEMPLATE = """Answer the following question as best you can. You have access to the following

{tools}


Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# STEP 3: Create and Run the Agent (Simulated)

def create_multihop_react_agent():
    """Create a ReAct agent setup (tools + prompt) for demonstration."""
    # Define tools list
    tools = [search_knowledge_base, calculator]

    # Create the prompt
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

    print("MULTI-HOP ReAct AGENT WITH LANGCHAIN")
    print("=" * 70)
    print("\nAgent Setup:")
    print(f"Tools: {[tool.name for tool in tools]}")
    print(f"Pattern: ReAct (Reasoning + Acting)")
    print(f"Multi-hop: Can chain multiple tool calls")
    print("\n" + "=" * 70 + "\n")

    return tools, prompt


def simulate_react_execution(question: str, tools: list):
    """
    Simulates a ReAct agent execution with detailed step-by-step output.
    This shows the Thought-Action-Observation loop clearly for learning.
    """
    print(f"QUESTION: {question}\n")
    print("=" * 70)

    # Multi-hop execution simulation
    steps = []

    if "capital" in question.lower() and "population" in question.lower():
        # Multi-hop question example
        steps = [
            {
                "step": 1,
                "thought": "I need to first find out what the capital of France is.",
                "action": "search_knowledge_base",
                "action_input": "capital of france",
                "observation": "Paris",
            },
            {
                "step": 2,
                "thought": "Now I know the capital is Paris. I need to find its population.",
                "action": "search_knowledge_base",
                "action_input": "population of paris",
                "observation": "2.1 million people",
            },
            {
                "step": 3,
                "thought": "I now have both pieces of information needed to answer the question.",
                "final_answer": "The capital of France is Paris, and its population is 2.1 million people.",
            },
        ]

    elif "eiffel tower" in question.lower() and "where" in question.lower():
        steps = [
            {
                "step": 1,
                "thought": "I need to find where the Eiffel Tower is located.",
                "action": "search_knowledge_base",
                "action_input": "eiffel tower location",
                "observation": "The Eiffel Tower is located in Paris",
            },
            {
                "step": 2,
                "thought": "Now I know it's in Paris. Let me find more details about Paris.",
                "action": "search_knowledge_base",
                "action_input": "paris location",
                "observation": "Paris is in northern France, on the Seine River",
            },
            {
                "step": 3,
                "thought": "I now have complete location information.",
                "final_answer": "The Eiffel Tower is located in Paris, which is in northern France on the S",
            },
        ]

    # Display the execution
    for step_data in steps:
        step_num = step_data["step"]
        print(f"\n--- STEP {step_num} ---\n")

        print(f"Thought: {step_data['thought']}")

        if "action" in step_data:
            print(f"Action: {step_data['action']}")
            print(f"Action Input: {step_data['action_input']}")

            # Actually execute the tool
            tool_obj = next((t for t in tools if t.name == step_data["action"]), None)
            if tool_obj:
                result = tool_obj.invoke(step_data["action_input"])
                print(f"Observation: {result}")

        if "final_answer" in step_data:
            print(f"Final Answer: {step_data['final_answer']}")
            print(f"{'=' * 70}")


# STEP 4: Demonstrate with Examples

def main():
    """Run the LangChain ReAct demonstration"""

    # Create agent components
    tools, prompt = create_multihop_react_agent()

    # Example 1: Multi-hop question
    print("EXAMPLE 1: Multi-Hop Question")
    print("=" * 70)
    question1 = "What is the capital of France and what is its population?"
    simulate_react_execution(question1, tools)

    # Example 2: Multi-hop question
    print("EXAMPLE 2: Another Multi-Hop Question")
    print("=" * 70)
    question2 = "What is the Eifel tower located and describe that location?"
    simulate_react_execution(question2, tools)

if __name__ == "__main__":
    main()