# Multi-Hop RAG Reasoning Chain with LangChain

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from typing import List, Dict
import re
# multi_hop_rag.py

import os
from dotenv import load_dotenv
from typing import List, TypedDict

from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS

# =========================
# 1. LOAD ENV + LLM
# =========================

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)



# embeddings = GoogleGenerativeAIEmbeddings(
#     model="text-embedding-001",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="embedding-001"
# )
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
#emb = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
#print(embeddings.embed_query("hello"))

# =========================
# 2. KNOWLEDGE BASE
# =========================

documents = [
    Document(
        page_content="France is a country in Western Europe. Its capital city is Paris.",
        metadata={"id": "doc1"}
    ),
    Document(
        page_content="Paris is located in northern France on the Seine River.",
        metadata={"id": "doc2"}
    ),
    Document(
        page_content="The population of Paris is approximately 2.1 million people in the city proper.",
        metadata={"id": "doc3"}
    ),
    Document(
        page_content="The Greater Paris metropolitan area has over 12 million inhabitants.",
        metadata={"id": "doc4"}
    ),
    Document(
        page_content="The Eiffel Tower is located in Paris and was completed in 1889.",
        metadata={"id": "doc5"}
    ),
    Document(
        page_content="Gustave Eiffel designed the Eiffel Tower for the 1889 World's Fair.",
        metadata={"id": "doc6"}
    ),
]


# =========================
# 3. VECTOR DATABASE
# =========================

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =========================
# 4. SCHEMAS
# =========================

class Decomposition(BaseModel):
    sub_questions: List[str]


class Answer(BaseModel):
    answer: str


# =========================
# 5. PLANNER (LLM DECOMPOSITION)
# =========================

planner_prompt = ChatPromptTemplate.from_template(
    """
You are a reasoning planner.

Break the following question into step-by-step sub-questions
that can be answered sequentially using retrieval.

Question:
{question}

Return only the sub-questions.
"""
)

planner_chain = (
    planner_prompt
    | model.with_structured_output(Decomposition)
)


# =========================
# 6. ANSWER GENERATOR
# =========================

answer_prompt = ChatPromptTemplate.from_template(
    """
You are a precise QA system.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

If the answer is not in the context, say "Not found".
"""
)

answer_chain = (
    answer_prompt
    | model.with_structured_output(Answer)
)


# =========================
# 7. MULTI-HOP RAG ENGINE
# =========================

class MultiHopRAG:

    def __init__(self):
        self.trace = []

    def retrieve(self, query: str):
        return retriever.invoke(query)

    def format_context(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def plan(self, question: str) -> List[str]:
        result = planner_chain.invoke({"question": question})
        return result.sub_questions

    def answer(self, question: str, context: str) -> str:
        result = answer_chain.invoke(
            {"question": question, "context": context}
        )
        return result.answer

    def run(self, question: str) -> str:

        print("\n🔷 MULTI-HOP RAG START\n")

        sub_questions = self.plan(question)

        print("Planned Steps:")
        for i, q in enumerate(sub_questions, 1):
            print(f"{i}. {q}")

        answers = []

        for i, sub_q in enumerate(sub_questions, 1):

            print(f"\n🔹 Hop {i}")
            print(f"Question: {sub_q}")

            # Replace placeholders if needed
            for j, ans in enumerate(answers, 1):
                sub_q = sub_q.replace(f"{{answer_{j}}}", ans)

            docs = self.retrieve(sub_q)
            context = self.format_context(docs)

            print("Retrieved Docs:")
            for d in docs:
                print("-", d.page_content)

            ans = self.answer(sub_q, context)

            print("Answer:", ans)

            answers.append(ans)

            self.trace.append({
                "hop": i,
                "question": sub_q,
                "answer": ans,
                "docs": [d.metadata.get("id") for d in docs]
            })

        final_answer = answers[-1] if answers else "No answer"

        print("\n🔷 FINAL ANSWER")
        print(final_answer)

        return final_answer


# =========================
# 8. RUN DEMO
# =========================

if __name__ == "__main__":

    rag = MultiHopRAG()

    question = "What is the population of the capital of France?"

    rag.run(question)