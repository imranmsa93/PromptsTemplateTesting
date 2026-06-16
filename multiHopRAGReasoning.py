# Multi-Hop RAG Reasoning Chain with LangChain

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from typing import List, Dict
import re

# STEP 1: Create Knowledge Base

KNOWLEDGE_BASE = [
    Document(
        "France is a country in Western Europe. Its capital city is Paris.",
        metadata={"id": "doc1", "topic": "geography"},
    ),
    Document(
        "Paris is located in northern France on the Seine River.",
        metadata={"id": "doc2", "topic": "geography"},
    ),
    Document(
        "The population of Paris is approximately 2.1 million people in the city proper.",
        metadata={"id": "doc3", "topic": "demographics"},
    ),
    Document(
        "The Greater Paris metropolitan area has over 12 million inhabitants.",
        metadata={"id": "doc4", "topic": "demographics"},
    ),
    Document(
        "The Eiffel Tower is located in Paris and was completed in 1889.",
        metadata={"id": "doc5", "topic": "landmarks"},
    ),
    Document(
        "The Eiffel Tower stands 330 meters tall including antennas.",
        metadata={"id": "doc6", "topic": "landmarks"},
    ),
    Document(
        "Gustave Eiffel designed the Eiffel Tower for the 1889 World's Fair.",
        metadata={"id": "doc7", "topic": "history"},
    ),
    Document(
        "Gustave Eiffel was born in 1832 in Dijon, France.",
        metadata={"id": "doc8", "topic": "biography"},
    ),
    Document(
        "Germany's capital is Berlin, which has a population of 3.7 million.",
        metadata={"id": "doc9", "topic": "geography"},
    ),
    Document(
        "The Seine River flows through Paris and is 777 kilometers long.",
        metadata={"id": "doc10", "topic": "geography"},
    ),
]

# STEP 2: Document Retriever

class SimpleRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def retrieve(self, query: str, k: int = 2) -> List[Document]:
        query_terms = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            if overlap > 0:
                score = overlap / len(query_terms)
                scored_docs.append((doc, score))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]


# STEP 3: Question Decomposition

DECOMPOSITION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Break down this complex question into simpler sub-questions.\n"
        "Each sub-question should be answerable independently and should chain.\n\n"
        "Question: {question}\n\nSub-questions:"
    ),
)


class QuestionDecomposer:
    @staticmethod
    def decompose(complex_question: str) -> List[str]:
        print(f"COMPLEX QUESTION: {complex_question}\n")
        print("DECOMPOSING INTO SUB-QUESTIONS...\n")
        q = complex_question.lower()
        sub_questions: List[str] = []

        if "population" in q and "capital" in q:
            sub_questions = [
                "What is the capital of France?",
                "What is the population of {{answer_1}}?",
            ]
        elif "where" in q and "designer" in q and "born" in q:
            sub_questions = [
                "Who designed the Eiffel Tower?",
                "Where was {{answer_1}} born?",
            ]
        elif "height" in q and "location" in q:
            sub_questions = [
                "Where is the Eiffel Tower located?",
                "How tall is the Eiffel Tower?",
            ]

        for i, sq in enumerate(sub_questions, 1):
            print(f"{i}. {sq}")
        print(f"\nDecomposed into {len(sub_questions)} sub-questions\n")
        return sub_questions


# STEP 4: Answer Extraction

EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "Based on the following context, answer the question concisely.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    ),
)


class AnswerExtractor:
    @staticmethod
    def extract(question: str, documents: List[Document]) -> str:
        context = " ".join(doc.page_content for doc in documents)
        q = question.lower()

        if "capital" in q and "france" in q and "paris" in context:
            return "Paris"
        if "population" in q:
            m = re.search(r"(\d+\.?\d*\s*million)", context)
            if m:
                return m.group(1)
        if "who designed" in q or "designer" in q:
            if "Gustave Eiffel" in context:
                return "Gustave Eiffel"
        if "where was" in q and "born" in q:
            m = re.search(r"born in (\d{4}) in ([^,\.]+)", context, re.IGNORECASE)
            if m:
                return f"{m.group(2)} in {m.group(1)}"
        if "where is" in q and "located" in q:
            if "Eiffel Tower is located in Paris" in context:
                return "Paris, France"
        if "tall" in q or "height" in q:
            m = re.search(r"(\d+\s*meters)", context)
            if m:
                return m.group(1)
        return "Unable to extract answer from context"


# STEP 5: Multi-Hop RAG Chain

class MultiHopRAGChain:
    def __init__(self, retriever: SimpleRetriever):
        self.retriever = retriever
        self.decomposer = QuestionDecomposer()
        self.extractor = AnswerExtractor()
        self.reasoning_trace: List[Dict] = []

    def execute_hop(
        self, hop_num: int, sub_question: str, previous_answers: List[str]
    ) -> Dict:
        print(f"HOP {hop_num}")
        print("=" * 60)

        current_question = sub_question
        for i, prev_answer in enumerate(previous_answers, 1):
            placeholder = f"{{{{answer_{i}}}}}"
            if placeholder in current_question:
                current_question = current_question.replace(placeholder, prev_answer)
                print(f"Substituting {placeholder} with '{prev_answer}'")

        print(f"\nSub-Question: {current_question}\n")

        retrieved_docs = self.retriever.retrieve(current_question, k=2)
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for doc in retrieved_docs:
            print(f"[{doc.metadata.get('id')}] {doc.page_content}")

        answer = self.extractor.extract(current_question, retrieved_docs)
        print(f"\nExtracted Answer: {answer}\n")

        hop_trace = {
            "hop": hop_num,
            "original_question": sub_question,
            "substituted_question": current_question,
            "retrieved_docs": [doc.metadata["id"] for doc in retrieved_docs],
            "answer": answer,
        }
        self.reasoning_trace.append(hop_trace)
        return hop_trace

    def invoke(self, complex_question: str) -> str:
        print("MULTI-HOP RAG CHAIN EXECUTION")
        print("=" * 60 + "\n")

        self.reasoning_trace = []
        sub_questions = self.decomposer.decompose(complex_question)
        if not sub_questions:
            return "Unable to decompose question"

        answers: List[str] = []
        for i, sub_q in enumerate(sub_questions, 1):
            hop_result = self.execute_hop(i, sub_q, answers)
            answers.append(hop_result["answer"])

        final_answer = answers[-1] if answers else "No answer found"

        print("FINAL ANSWER")
        print("=" * 60)
        print(f"\nQuestion: {complex_question}")
        print(f"Answer: {final_answer}\n")

        return final_answer

    def display_trace(self):
        print("REASONING TRACE")
        print("=" * 60 + "\n")
        for trace in self.reasoning_trace:
            print(f"Hop {trace['hop']}:")
            print(f"  Template: {trace['original_question']}")
            print(f"  Actual:   {trace['substituted_question']}")
            print(f"  Docs:     {', '.join(trace['retrieved_docs'])}")
            print(f"  Answer:   {trace['answer']}\n")


# MAIN DEMONSTRATION

def main():
    retriever = SimpleRetriever(KNOWLEDGE_BASE)
    rag_chain = MultiHopRAGChain(retriever)

    print("\nExample 1: Population of the capital of France\n")
    q1 = "What is the population of the capital of France?"
    rag_chain.invoke(q1)
    rag_chain.display_trace()

    print("\nExample 2: Birthplace of the designer of the Eiffel Tower\n")
    q2 = "Where was the designer of the Eiffel Tower born?"
    rag_chain.invoke(q2)
    rag_chain.display_trace()


if __name__ == "__main__":
    main()