# Adding Verification to ReAct Tool-Chaining with LangChain

from langchain_core.tools import tool
from typing import Any, Dict, List
import json

# STEP 1: Define Tools with LangChain

@tool
def search_price_database(query: str) -> str:
    """Search for product prices and discount information."""
    database = {
        "laptop price": "799",
        "laptop discount": "20",  # 20% discount
        "tax rate": "8",          # 8% tax
        "shipping cost": "25",
    }

    query_lower = query.lower()
    for key, value in database.items():
        if key in query_lower:
            return value
    return "Not found"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations. Input must be a valid Python math expression."""
    try:
        # Safe eval with limited namespace
        result = eval(expression, {"__builtins__": {}}, {})
        return str(round(result, 2))
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def verify_result(value: str, context: str) -> str:
    """
    Verification tool - validates if a result is reasonable given the context.

    Args:
        value: The value to verify.
        context: Description of what this value represents (e.g., 'laptop price', 'discount percentage').

    Returns:
        JSON string with verification result: { "is_valid": bool, "checks": [...], "value": str }
    """
    checks = []
    is_valid = True

    try:
        numeric_value = float(value)
        checks.append({"check": "Is numeric", "passed": True})

        # Context-specific validation
        if "price" in context.lower() or "cost" in context.lower():
            if numeric_value < 0:
                checks.append({"check": "Non-negative value", "passed": False})
                is_valid = False
            else:
                checks.append({"check": "Non-negative value", "passed": True})

            # Range check for prices
            if "laptop" in context.lower():
                if 50 <= numeric_value <= 5000:
                    checks.append(
                        {"check": "Reasonable laptop price range", "passed": True}
                    )
                else:
                    checks.append(
                        {"check": "Reasonable laptop price range", "passed": False}
                    )
                    is_valid = False

        elif "discount" in context.lower() or "tax" in context.lower():
            if 0 <= numeric_value <= 100:
                checks.append({"check": "Valid percentage (0-100)", "passed": True})
            else:
                checks.append({"check": "Valid percentage (0-100)", "passed": False})
                is_valid = False

    except ValueError:
        checks.append({"check": "Is numeric", "passed": False})
        is_valid = False

    return json.dumps({"is_valid": is_valid, "checks": checks, "value": value})


# STEP 2: Verified ReAct Agent with LangChain

class VerifiedReActAgent:
    def __init__(self, tools: List):
        self.tools = {tool.name: tool for tool in tools}
        self.verified_facts = {}
        self.execution_log = []

    def execute_with_verification(self, question: str):
        print(f"QUESTION: {question}")
        print(f"{'=' * 70}\n")

        # Define reasoning steps for the question
        reasoning_steps = self._plan_steps(question)

        for step_num, step_plan in enumerate(reasoning_steps, 1):
            print(f"STEP {step_num}: {step_plan['description']}")
            print(f"{'=' * 70}\n")

            # THOUGHT
            print(f"Thought: {step_plan['thought']}")

            # ACTION
            print(f"Action: {step_plan['tool_name']}")
            print(f"Action Input: {step_plan['tool_input']}")

            # Execute the tool
            tool = self.tools[step_plan["tool_name"]]
            observation = tool.invoke(step_plan["tool_input"])

            # OBSERVATION
            print(f"Observation: {observation}")

            # VERIFICATION (Key addition!)
            print(f"\nVERIFICATION STEP:")
            verification_needed = step_plan.get("verify", True)

            if verification_needed and step_plan["tool_name"] != "verify_result":
                verified = self._verify_observation(
                    observation, step_plan["context"], step_plan["fact_name"]
                )

                if not verified:
                    print(f"\nVerification failed! Stopping execution.\n")
                    return None
            else:
                print(f"   Skipping verification for this step")

            print()

        # Final answer
        print(f"FINAL ANSWER")
        print(f"{'=' * 70}")
        final_price = self.verified_facts.get("final_price", 0)
        print(f"The total cost of the laptop is: ${final_price:.2f}\n")

        return final_price

    def _verify_observation(self, observation: str, context: str, fact_name: str) -> bool:
        """Verify an observation before storing it as a fact"""
        print(f"   Verifying: '{observation}' as {fact_name}")

        # Use the verification tool
        verify_tool = self.tools["verify_result"]
        # FIX: pass arguments as a dict matching the tool signature
        verification_json = verify_tool.invoke(
            {"value": observation, "context": context}
        )

        # Parse verification result
        result = json.loads(verification_json)

        # Display checks
        for check in result["checks"]:
            status = "OK" if check["passed"] else "FAIL"
            print(f"   [{status}] {check['check']}")

        if result["is_valid"]:
            print(f"   Verification PASSED - Storing as verified fact")
            self.verified_facts[fact_name] = float(observation)
            return True
        else:
            print(f"   Verification FAILED - Not storing this result")
            return False

    def _plan_steps(self, question: str) -> List[Dict]:
        """Plan the reasoning steps for the question"""
        if "laptop" in question.lower() and "total cost" in question.lower():
            return [
                {
                    "description": "Get base laptop price",
                    "thought": "I need to find the base price of the laptop",
                    "tool_name": "search_price_database",
                    "tool_input": "laptop price",
                    "context": "laptop base price",
                    "fact_name": "base_price",
                    "verify": True,
                },
                {
                    "description": "Get discount rate",
                    "thought": "I need to find the discount percentage",
                    "tool_name": "search_price_database",
                    "tool_input": "laptop discount",
                    "context": "discount percentage",
                    "fact_name": "discount",
                    "verify": True,
                },
                {
                    "description": "Calculate price after discount",
                    "thought": "I should calculate the price after applying the discount",
                    "tool_name": "calculate",
                    "tool_input": f"{self.verified_facts.get('base_price', 799)} * (1 - {self.verified_facts.get('discount', 20)} / 100)",
                    "context": "laptop price after discount",
                    "fact_name": "discounted_price",
                    "verify": True,
                },
                {
                    "description": "Get tax rate",
                    "thought": "I need to find the tax rate to add to the price",
                    "tool_name": "search_price_database",
                    "tool_input": "tax rate",
                    "context": "tax rate percentage",
                    "fact_name": "tax_rate",
                    "verify": True,
                },
                {
                    "description": "Calculate final price with tax",
                    "thought": "I should calculate the final price including tax",
                    "tool_name": "calculate",
                    "tool_input": f"{self.verified_facts.get('discounted_price', 639.2)} * (1 + {self.verified_facts.get('tax_rate', 8)} / 100)",
                    "context": "final laptop price with tax",
                    "fact_name": "final_price",
                    "verify": True,
                },
            ]
        return []


# STEP 3: Run the verification demonstration

def main():
    """Run the verification demonstration"""

    print("ReAct WITH VERIFICATION - LANGCHAIN")
    print("=" * 70)

    # Create tools
    tools = [search_price_database, calculate, verify_result]

    print("\nAvailable Tools:")
    for tool in tools:
        print(f"   • {tool.name}: {tool.description}")

    # Create verified agent
    agent = VerifiedReActAgent(tools)

    # Run example
    print("EXAMPLE: Multi-Step Calculation with Verification")
    print("=" * 70)

    question = "What is the total cost of a laptop with 20% discount including 8% tax?"
    result = agent.execute_with_verification(question)

    # Show verified facts
    print("VERIFIED FACTS ACCUMULATED")
    print(f"{'=' * 70}")
    for fact_name, value in agent.verified_facts.items():
        print(f"   • {fact_name}: ${value:.2f}")


if __name__ == "__main__":
    main()