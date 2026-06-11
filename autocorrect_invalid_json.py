import os
import json
import time
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from colorama import init, Fore, Style
from tqdm import tqdm


# ---------- Init colorama ----------

init(autoreset=True)


# ---------- Pydantic schemas ----------

class DailyScheduleItem(BaseModel):
    day_number: int
    focus_topic: str
    estimated_hours: float
    tasks: List[str]


class StudyPlan(BaseModel):
    student_name: str
    goal: str
    duration_days: int
    topics: List[str]
    daily_schedule: List[DailyScheduleItem]


# ---------- Styled print helpers ----------

def print_header(text: str):
    print(Fore.CYAN + Style.BRIGHT + text + Style.RESET_ALL)


def print_step(text: str):
    print(Fore.YELLOW + text + Style.RESET_ALL)


def print_success(text: str):
    print(Fore.GREEN + text + Style.RESET_ALL)


def print_error(text: str):
    print(Fore.RED + text + Style.RESET_ALL)


# ---------- Model + Chains ----------

def build_model() -> ChatGoogleGenerativeAI:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
    )


def build_raw_plan_chain(model: ChatGoogleGenerativeAI):
    """
    Chain that produces a study plan JSON.
    Sometimes it may still be invalid or not exactly match the schema.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a study coach that produces JSON study plans for a student.\n"
                "You MUST return STRICT JSON with the following structure:\n"
                "- student_name: string\n"
                "- goal: string\n"
                "- duration_days: integer\n"
                "- topics: array of strings\n"
                "- daily_schedule: array of objects, each with:\n"
                "    - day_number: integer (starting from 1)\n"
                "    - focus_topic: string\n"
                "    - estimated_hours: number\n"
                "    - tasks: array of strings\n"
                "Return ONLY JSON, no explanation."
            ),
            (
                "human",
                "Create a 10-day study plan JSON for this student:\n\n"
                "{student_description}"
            ),
        ]
    )
    return prompt | model | StrOutputParser()


def build_fix_json_chain(model: ChatGoogleGenerativeAI):
    """
    Chain that receives broken or schema-mismatched JSON and fixes it.
    NOTE: No literal { } blocks here to avoid LangChain template confusion.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You fix invalid or schema-mismatched JSON for study plans.\n"
                "You MUST output a single valid JSON object with exactly these keys:\n"
                "- student_name: string\n"
                "- goal: string\n"
                "- duration_days: integer\n"
                "- topics: array of strings\n"
                "- daily_schedule: array of objects\n"
                "Each object inside daily_schedule MUST have:\n"
                "- day_number: integer\n"
                "- focus_topic: string\n"
                "- estimated_hours: number\n"
                "- tasks: array of strings\n"
                "Important rules:\n"
                "- daily_schedule MUST be an array (not weekday/weekend objects).\n"
                "- Include one entry per day, covering the entire plan duration.\n"
                "- Output ONLY raw JSON, with no backticks and no explanation text."
            ),
            (
                "human",
                "Fix this JSON so it matches the required schema:\n\n{bad_json}"
            ),
        ]
    )
    return prompt | model | StrOutputParser()


# ---------- Strip Markdown Code Fences ----------

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------- Core JSON Parsing Logic ----------

def parse_or_fix_with_llm(raw_text: str, fixer_chain) -> StudyPlan:
    print_step("\n--- Trying to parse JSON directly ---")
    try:
        cleaned = strip_markdown_fences(raw_text)
        data = json.loads(cleaned)
        plan = StudyPlan(**data)
        print_success("Parsing succeeded without fixing.")
        return plan
    except (json.JSONDecodeError, ValidationError) as e:
        print_error(f"Initial parsing failed: {repr(e)}")

    print_step("\n--- Asking LLM to auto-correct JSON ---")

    # Visual progress bar to show "work"
    for _ in tqdm(range(30), desc="Auto-correcting JSON", ncols=70):
        time.sleep(0.03)

    fixed_json = fixer_chain.invoke({"bad_json": raw_text})
    print(Fore.MAGENTA + "\nFixed JSON:\n" + Style.RESET_ALL + fixed_json)

    print_step("\n--- Parsing fixed JSON into StudyPlan ---")
    cleaned_fixed = strip_markdown_fences(fixed_json)
    data = json.loads(cleaned_fixed)
    plan = StudyPlan(**data)
    print_success("Parsing succeeded after fixing.")
    return plan


# ---------- Pretty Printing ----------

def pretty_print_study_plan(plan: StudyPlan):
    print_header("\n==================== FORMATTED STUDY PLAN ====================")
    print(Fore.WHITE + f"Student Name : {plan.student_name}")
    print(Fore.WHITE + f"Goal         : {plan.goal}")
    print(Fore.WHITE + f"Duration     : {plan.duration_days} days")
    print(Fore.WHITE + f"Topics       : {', '.join(plan.topics)}")
    print(Fore.WHITE + "\nDaily Schedule")
    print(Fore.WHITE + "--------------------------------------------------------------")

    for day in plan.daily_schedule:
        print(Fore.CYAN + f"Day {day.day_number}: {day.focus_topic}")
        print(Fore.WHITE + f"  Hours: {day.estimated_hours}")
        print(Fore.WHITE + "  Tasks:")
        for task in day.tasks:
            print(Fore.WHITE + f"    • {task}")
        print(Fore.WHITE + "--------------------------------------------------------------")


# ---------- MAIN ----------

def main():
    print_header("\n===  Auto-Correcting Invalid JSON Outputs ===")

    model = build_model()
    raw_chain = build_raw_plan_chain(model)
    fixer_chain = build_fix_json_chain(model)

    student_description = (
        "My name is Priya. I have 10 days to prepare for a data structures and algorithms "
        "coding interview. I know basic Python and can study about 2 hours on weekdays and "
        "4 hours on weekends. Focus on arrays, strings, linked lists, trees, and dynamic programming."
    )

    print_step("\n--- Asking LLM for JSON study plan ---")
    raw_json = raw_chain.invoke({"student_description": student_description})
    print(Fore.MAGENTA + "Raw Output:\n" + Style.RESET_ALL + raw_json)

    # Intentionally corrupt the JSON to simulate invalid output (missing quotes around duration_days)
    cleaned_raw = strip_markdown_fences(raw_json)
    broken_json = cleaned_raw.replace('"duration_days"', "duration_days", 1)

    plan = parse_or_fix_with_llm(broken_json, fixer_chain)

    print_step("\n=== Final Parsed StudyPlan (Raw Pydantic) ===")
    print(Fore.WHITE + str(plan))

    pretty_print_study_plan(plan)


if __name__ == "__main__":
    main()
