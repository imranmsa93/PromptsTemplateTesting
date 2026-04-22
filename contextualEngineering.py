from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0.7
)

print("Contextual Engineering \n")

# Collect Context from the User
topic = input("Enter the main topic you want to explain: ")
system_role = input("Define AI role (e.g., expert teacher, mentor): ")
audience = input("Target audience (beginner / student / professional): ")
tone = input("Tone (friendly / formal / expert): ")
previous_context = input("Previous conversation context (optional): ")
external_info = input("Extra domain knowledge (optional): ")
format_instructions = input("Output format (optional, e.g., JSON / bullets): ")

print("\nGenerating enhanced output using Context Engineering...\n")

# Build Context Engineering Prompt
prompt_template = """
You are {system_role}.
Audience Type: {audience}
Tone Style: {tone}

Previous Context:
{previous_context}

External Knowledge:
{external_info}

Formatting Instructions:
{format_instructions}

Your goal:
Explain the following topic clearly and accurately using principles of Context Engineering.

Topic: "{topic}"

Now produce the best answer possible:
"""

prompt = PromptTemplate(
    input_variables=[
        "system_role",
        "audience",
        "tone",
        "previous_context",
        "external_info",
        "format_instructions",
        "topic"
    ],
    template=prompt_template,
)

final_prompt = prompt.format(
    system_role=system_role,
    audience=audience,
    tone=tone,
    previous_context=previous_context or "None provided",
    external_info=external_info or "None provided",
    format_instructions=format_instructions or "Free text",
    topic=topic
)

#print the final prompt before invoking the llm
print(final_prompt);
#call the mode 
response = llm.invoke(final_prompt)

#Show output
print("\nGemini Response:\n", response.content)