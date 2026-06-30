from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import List, Dict, Tuple
# Knowledge Base and Metadata

@dataclass
class Doc:
    id: str
    text: str
    topic: str
    updated: str     # YYYY-MM-DD
    priority: int    # 1..5
    tags: List[str]


KB: List[Doc] = [
    Doc(
        id="invite_01",
        topic="team",
        updated="2025-10-01",
        priority=5,
        tags=["invite", "teammate", "member", "members", "role", "roles"],
        text="Invite teammates: Settings > Team > Members > Invite. Enter emails, choose a role (Admin/Member/Viewer), then send. Invites expire in 7 days; Admins can resend or revoke.",
    ),
    Doc(
        id="roles_01",
        topic="team",
        updated="2025-08-20",
        priority=4,
        tags=["role", "roles", "admin", "member", "viewer", "permissions"],
        text="Roles: Admin manages billing/security/members. Member can create/edit projects. Viewer is read-only. Role changes apply immediately.",
    ),
    Doc(
        id="sso_01",
        topic="security",
        updated="2025-09-10",
        priority=3,
        tags=["sso", "saml", "enterprise", "security"],
        text="SSO: Enterprise only. Configure in Settings > Security > SSO. Supports SAML 2.0. Admins can enforce SSO for all users after verification.",
    ),
    Doc(
        id="billing_01",
        topic="billing",
        updated="2025-06-15",
        priority=2,
        tags=["billing", "invoice", "payment", "upgrade"],
        text="Billing: Update payment method in Settings > Billing. Invoices are emailed monthly to billing admins. Proration may apply when upgrading mid-cycle.",
    ),
    Doc(
        id="projects_01",
        topic="projects",
        updated="2025-07-05",
        priority=2,
        tags=["project", "projects", "templates", "archive", "archived"],
        text="Projects: Create a project from the dashboard using New Project. Use templates for common workflows. Archived projects become read-only.",
    ),
]

#  Retrieval and Metadata Ranking
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "in", "on", "for", "of", "with", "my", "i",
    "is", "are", "do", "does", "how", "what", "when", "where", "can", "could", "please"
}

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None

def freshness_score(updated: str) -> float:
    """0..1; newer = higher"""
    dt = parse_date(updated)
    if not dt:
        return 0.0
    days = max((datetime.now() - dt).days, 0)
    return math.exp(-days / 180.0)  # ~6-month decay

def retrieve(query: str, k: int = 3) -> List[Tuple[Doc, float]]:
    q_tokens = set(tokenize(query))

    scored: List[Tuple[Doc, float]] = []
    for doc in KB:
        d_tokens = set(tokenize(doc.text)) | set([t.lower() for t in doc.tags])

        # lexical overlap (0..1-ish)
        overlap = len(q_tokens & d_tokens)
        base = overlap / max(len(q_tokens), 1)

        # metadata boosts
        pr_boost = 0.08 * min(max(doc.priority, 1), 5)           # 0.08..0.40
        fr_boost = 0.35 * freshness_score(doc.updated)           # 0..0.35
        tag_boost = 0.15 if any(t in q_tokens for t in map(str.lower, doc.tags)) else 0.0

        score = base + pr_boost + fr_boost + tag_boost
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

def format_context(docs: List[Tuple[Doc, float]]) -> str:
    blocks = []
    for doc, score in docs:
        blocks.append(
            f"[{doc.id}] topic={doc.topic} updated={doc.updated} priority={doc.priority} score={score:.2f}\n"
            f"{doc.text}"
        )
    return "\n\n".join(blocks)

#  “LLM-like” Draft / Review / Refine 

def draft_answer(question: str, docs: List[Tuple[Doc, float]]) -> str:

    if not docs:
        return "I don't have enough context to answer. Which TaskFlow area is this about (team, billing, security, projects)?"

    top_doc, top_score = docs[0]
    if top_score < 0.35:
        return ("I may be missing the right documentation to answer that confidently.\n"
                "Can you clarify what you’re trying to do (invite teammates, change roles, enable SSO, billing, or projects)?")

    # Build a “support-style” response using the best matching snippet
    if "invite" in question.lower() or "teammate" in question.lower() or "member" in question.lower():
        return (
            "To invite teammates in TaskFlow:\n"
            "1) Open **Settings**\n"
            "2) Go to **Team → Members → Invite**\n"
            "3) Enter their email(s)\n"
            "4) Choose a role (**Admin / Member / Viewer**)\n"
            "5) Send the invite\n\n"
            "Notes: Invites expire in **7 days**. Admins can **resend** or **revoke** invites."
        )

    if "role" in question.lower() or "permission" in question.lower():
        return (
            "TaskFlow roles work like this:\n"
            "- **Admin**: manages billing, security, and members\n"
            "- **Member**: creates/edits projects\n"
            "- **Viewer**: read-only access\n\n"
            "Role changes apply immediately."
        )

    if "sso" in question.lower() or "saml" in question.lower():
        return (
            "SSO setup in TaskFlow:\n"
            "1) Go to **Settings → Security → SSO**\n"
            "2) Configure **SAML 2.0**\n"
            "3) Verify, then (optional) enforce SSO for all users\n\n"
            "Note: SSO is available on **Enterprise** plans."
        )

    if "billing" in question.lower() or "invoice" in question.lower():
        return (
            "Billing in TaskFlow:\n"
            "1) Go to **Settings → Billing**\n"
            "2) Update your payment method\n\n"
            "Invoices are emailed monthly to billing admins. Proration may apply when upgrading mid-cycle."
        )

    # Fallback: quote best doc 
    return f"Here's what I found:\n- {top_doc.text}"

def review_answer(question: str, answer: str, docs: List[Tuple[Doc, float]]) -> Dict:

    top_score = docs[0][1] if docs else 0.0

    # If answer indicates missing context → RETRY
    if "clarify" in answer.lower() or "missing" in answer.lower() or "don’t have enough" in answer.lower():
        return {
            "verdict": "RETRY",
            "issues": ["Insufficient context retrieved"],
            "improved_query": refine_query_heuristic(question),
        }

    # If retrieval is weak → RETRY
    if top_score < 0.35:
        return {
            "verdict": "RETRY",
            "issues": ["Low retrieval confidence"],
            "improved_query": refine_query_heuristic(question),
        }

    # If answer doesn't seem to reference typical UI paths when it should → RETRY
    if ("how" in question.lower() or "steps" in question.lower()) and ("settings" not in answer.lower()):
        return {
            "verdict": "RETRY",
            "issues": ["Answer not actionable (missing navigation steps)"],
            "improved_query": refine_query_heuristic(question),
        }

    return {"verdict": "PASS", "issues": [], "improved_query": ""}

def refine_query_heuristic(question: str) -> str:

    q = question.lower()
    base = " ".join(tokenize(question))[:200]

    if any(w in q for w in ["invite", "teammate", "member", "members"]):
        return base + " settings team members invite roles admin viewer"
    if any(w in q for w in ["role", "roles", "permissions"]):
        return base + " roles permissions admin member viewer team settings"
    if any(w in q for w in ["sso", "saml"]):
        return base + " settings security sso saml enterprise"
    if any(w in q for w in ["billing", "invoice", "payment"]):
        return base + " settings billing invoice payment proration"
    return base + " settings steps guide"

# 4) Self-Correcting Orchestrator

def self_correcting_answer(question: str, rounds: int = 2, k: int = 3) -> Tuple[str, List[Dict]]:
    trace = []
    query = question

    for r in range(1, rounds + 1):
        docs = retrieve(query, k=k)
        ctx = format_context(docs)
        ans = draft_answer(question, docs)
        review = review_answer(question, ans, docs)

        trace.append({
            "round": r,
            "query": query,
            "top_docs": [d.id for d, _ in docs],
            "top_scores": [round(s, 2) for _, s in docs],
            "review": review
        })

        if review["verdict"] == "PASS":
            return ans, trace

        query = review["improved_query"] or refine_query_heuristic(question)

    # If still not passing, return best attempt and note
    return ans + "\n\n( Final note: this answer may be incomplete due to limited KB coverage.)", trace

#  CLI Runner

def main():
    print("\nSelf-Correcting Knowledge Query Pipeline")
    print("Type a question. Type /exit to quit.")

    last_trace = None

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"/exit", "/quit"}:
            break
        if user.lower() == "/trace":
            if not last_trace:
                print("\n(No trace yet — ask a question first.)\n")
            else:
                print("\n--- TRACE (Self-Correction) ---")
                for step in last_trace:
                    print(f"Round {step['round']}: query='{step['query']}'")
                    print(f"  top_docs={step['top_docs']}")
                    print(f"  top_scores={step['top_scores']}")
                    print(f"  review={step['review']}")
                print("------------------------------\n")
            continue

        answer, last_trace = self_correcting_answer(user, rounds=2, k=3)
        print("\nAssistant:\n" + answer + "\n")


if __name__ == "__main__":
    main()
