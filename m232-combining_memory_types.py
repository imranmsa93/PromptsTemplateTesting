import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Global policies ----------

GLOBAL_POLICIES = """\
- Never invent internal URLs, credentials, or secrets.
- Prefer safe, incremental deployment changes.
- If unsure about internal details, say so explicitly.
""".strip()

# ---------- Adaptive profile memory ----------

class ProfileMemory:
    def __init__(self) -> None:
        self.domains: set[str] = set()
        self.tech_stack: set[str] = set()

    def update_from_question(self, q: str) -> None:
        q = q.lower()

        if any(k in q for k in ["deployment", "deploy", "kubernetes", "ci/cd", "pipeline", "gitops", "infrastructure"]):
            self.domains.add("DevOps / Platform Engineering")

        tech_keywords = {
            "kubernetes": "Kubernetes",
            "docker": "Docker",
            "terraform": "Terraform",
            "aws": "AWS",
            "azure": "Azure",
            "gcp": "GCP",
            "python": "Python",
            "fastapi": "FastAPI",
            "sql": "SQL",
        }
        for k, label in tech_keywords.items():
            if k in q:
                self.tech_stack.add(label)

    def is_relevant(self, q: str) -> bool:
        q = q.lower()
        chit_chat = ["paris", "movie", "music", "food", "vacation", "travel", "game"]
        if any(w in q for w in chit_chat):
            return False
        return bool(self.domains or self.tech_stack)

    def format_for_question(self, q: str) -> str:
        if not self.is_relevant(q):
            return "(profile not used for this question)"
        lines = []
        if self.domains:
            lines.append(f"- Domains: {', '.join(sorted(self.domains))}")
        if self.tech_stack:
            lines.append(f"- Tech stack hints: {', '.join(sorted(self.tech_stack))}")
        return "\n".join(lines) if lines else "(profile not used for this question)"

PROFILE = ProfileMemory()

# ---------- Conversation memory ----------

class ConversationMemory:
    def __init__(self, max_turns: int = 6) -> None:
        self.turns: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def as_text(self) -> str:
        if not self.turns:
            return "(no prior conversation)"
        return "\n".join(
            f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
            for t in self.turns
        )

CONVO = ConversationMemory()

# ---------- Vector memory (Project Atlas) ----------

ATLAS_DOCS = [
    "Project Atlas is our internal platform for orchestrating deployment pipelines.",
    "Atlas uses a GitOps workflow where changes to the main branch trigger automated deployments.",
    "The Atlas API is implemented in Python using FastAPI, backed by a PostgreSQL database.",
    "Atlas runs on Kubernetes with separate clusters for staging and production workloads.",
    "A key performance issue has been long-running migrations on large production tables.",
    "Upcoming roadmap items include canary deployments, progressive traffic shifting, and RBAC improvements.",
]

def build_llm_and_vs():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set in .env")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    docs = [Document(page_content=text) for text in ATLAS_DOCS]
    vs = FAISS.from_documents(docs, emb)
    return llm, vs

def get_vector_context(vs: FAISS, question: str, max_distance: float = 0.6):
    docs_scores = vs.similarity_search_with_score(question, k=4)
    if not docs_scores:
        return "(vector layer not used)"

    best_dist = docs_scores[0][1]
    if best_dist > max_distance:
        return "(vector layer not used)"

    lines = []
    for doc, dist in docs_scores:
        if dist <= best_dist + 0.1:
            lines.append(f"- [{dist:.3f}] {doc.page_content}")
    return "\n".join(lines) if lines else "(vector layer not used)"

# ---------- Prompt & chain ----------

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for technical questions about deployments and Project Atlas.\n"
            "You receive three context layers:\n\n"
            "GLOBAL POLICIES:\n{policies}\n\n"
            "SESSION CONTEXT:\n"
            "  PROFILE (may say 'not used'):\n{profile}\n\n"
            "  RECENT CONVERSATION:\n{conversation}\n\n"
            "VECTOR KNOWLEDGE (Atlas docs, if relevant):\n{vector}\n\n"
            "Use policies first, then session context, then vector docs if relevant.\n"
            "If something is not covered, say so and use general knowledge.",
        ),
        ("human", "User question: {question}"),
    ]
)

def print_header():
    print("=" * 80)
    print(" Hybrid Memory Demo (concise) – Adaptive Profile + Multi-Layer Context")
    print("=" * 80)

def main():
    print_header()
    try:
        llm, vs = build_llm_and_vs()
    except Exception as e:
        print("Init failed:", e)
        return

    chain = (
        {
            "policies": lambda x: x["policies"],
            "profile": lambda x: x["profile"],
            "conversation": lambda x: x["conversation"],
            "vector": lambda x: x["vector"],
            "question": lambda x: x["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        # update memories
        PROFILE.update_from_question(q)
        CONVO.add("user", q)

        profile_block = PROFILE.format_for_question(q)
        convo_block = CONVO.as_text()
        vector_block = get_vector_context(vs, q)

        ctx = {
            "policies": GLOBAL_POLICIES,
            "profile": profile_block,
            "conversation": convo_block,
            "vector": vector_block,
            "question": q,
        }

        answer = chain.invoke(ctx)
        CONVO.add("assistant", answer)

        print("\n[Context Snapshot]")
        print("PROFILE:\n", profile_block)
        print("\nVECTOR:\n", vector_block)
        print("\n[Answer]")
        print(answer)
        print("-" * 80)

if __name__ == "__main__":
    main()
