from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from collections import defaultdict
from typing import List, Dict
import re

# STEP 1: Create Document Store (Simulated Vector Store)

class SimpleDocumentStore:
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        query_terms = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            if overlap > 0:
                score = overlap / len(query_terms)
                scored_docs.append((doc, score))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]


DOCUMENTS = [
    Document(
        page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        metadata={"id": "doc1"},
    ),
    Document(
        page_content="Deep learning uses neural networks with multiple layers to learn from data.",
        metadata={"id": "doc2"},
    ),
    Document(
        page_content="Supervised learning requires labeled training data to make predictions.",
        metadata={"id": "doc3"},
    ),
    Document(
        page_content="Artificial intelligence enables computers to perform tasks requiring human intelligence.",
        metadata={"id": "doc4"},
    ),
    Document(
        page_content="Neural networks are inspired by biological neurons in the human brain.",
        metadata={"id": "doc5"},
    ),
    Document(
        page_content="Training algorithms requires large datasets and computational resources.",
        metadata={"id": "doc6"},
    ),
    Document(
        page_content="AI models learn patterns from data to make intelligent decisions.",
        metadata={"id": "doc7"},
    ),
    Document(
        page_content="Deep neural networks can recognize images and understand natural language.",
        metadata={"id": "doc8"},
    ),
]

# STEP 2: Query Expansion with LangChain Prompt

QUERY_EXPANSION_PROMPT = PromptTemplate(
    input_variables=["original_query"],
    template="""You are an AI assistant that generates multiple search query variations.
Given an original query, generate 3 different variations that could help find relevant information.

Original Query: {original_query}

Generate variations using these strategies:
1. Synonym replacement
2. More specific phrasing
3. Related concepts

Return only the queries, one per line.""",
)


def generate_query_variations_simple(original_query: str) -> List[str]:
    print(f"ORIGINAL QUERY: '{original_query}'\n")
    print("Generating query variations...\n")
    variations = [original_query]
    lower_q = original_query.lower()
    if "machine learning" in lower_q:
        variations.extend(
            [
                "ML algorithms and training processes",
                "artificial intelligence learning systems",
                "neural networks and data-driven models",
            ]
        )
    elif "how" in lower_q and "work" in lower_q:
        variations.extend(
            [
                original_query.replace("how does", "explain the process of"),
                original_query.replace("work", "function and operate"),
                "training and inference in AI systems",
            ]
        )

    for i, var in enumerate(variations, 1):
        print(f"{i}. {var}")
    print()
    return variations


# STEP 3: Multi-Query Retrieval

class MultiQueryRetrieverDemo:
    def __init__(self, document_store: SimpleDocumentStore):
        self.document_store = document_store

    def retrieve_for_all_queries(
        self, queries: List[str], top_k: int = 3
    ) -> Dict[str, List[Document]]:
        print("MULTI-QUERY RETRIEVAL\n")
        all_results = {}
        for i, query in enumerate(queries, 1):
            print(f"Query {i}: '{query}'")
            results = self.document_store.similarity_search(query, k=top_k)
            print(f"  Retrieved {len(results)} documents:")
            for rank, doc in enumerate(results, 1):
                doc_id = doc.metadata.get("id", "unknown")
                print(f"    {rank}. [{doc_id}] {doc.page_content[:60]}...")
            print()
            all_results[query] = results
        return all_results


# STEP 4: Fusion Strategies

class FusionStrategies:
    @staticmethod
    def reciprocal_rank_fusion(
        all_results: Dict[str, List[Document]], k: int = 60
    ) -> List[tuple]:
        fused_scores = defaultdict(float)
        doc_map = {}
        for docs in all_results.values():
            for rank, doc in enumerate(docs, 1):
                doc_id = doc.metadata["id"]
                rrf_score = 1.0 / (k + rank)
                fused_scores[doc_id] += rrf_score
                doc_map[doc_id] = doc
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_map[doc_id]) for doc_id, score in ranked]

    @staticmethod
    def simple_voting(all_results: Dict[str, List[Document]]) -> List[tuple]:
        vote_counts = defaultdict(int)
        doc_map = {}
        for docs in all_results.values():
            for doc in docs:
                doc_id = doc.metadata["id"]
                vote_counts[doc_id] += 1
                doc_map[doc_id] = doc
        ranked = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, count, doc_map[doc_id]) for doc_id, count in ranked]

    @staticmethod
    def weighted_fusion(
        all_results: Dict[str, List[Document]], query_weights: List[float]
    ) -> List[tuple]:
        fused_scores = defaultdict(float)
        doc_map = {}
        for (query, docs), weight in zip(all_results.items(), query_weights):
            for rank, doc in enumerate(docs, 1):
                doc_id = doc.metadata["id"]
                rank_score = 1.0 / rank
                fused_scores[doc_id] += weight * rank_score
                doc_map[doc_id] = doc
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_map[doc_id]) for doc_id, score in ranked]


# STEP 5: Display Results

def display_final_ranking(
    ranked_results: List[tuple], strategy_name: str, top_n: int = 5
):
    print(f"\nFINAL RANKING - {strategy_name}\n")
    for rank, (doc_id, score, doc) in enumerate(ranked_results[:top_n], 1):
        print(f"{rank}. {doc_id} (score: {score:.4f})")
        print(f"   {doc.page_content}\n")


# MAIN DEMONSTRATION

def main():
    doc_store = SimpleDocumentStore(DOCUMENTS)
    multi_retriever = MultiQueryRetrieverDemo(doc_store)
    user_query = "How does machine learning work?"
    query_variations = generate_query_variations_simple(user_query)
    all_results = multi_retriever.retrieve_for_all_queries(query_variations, top_k=3)

    rrf_results = FusionStrategies.reciprocal_rank_fusion(all_results)
    display_final_ranking(rrf_results, "Reciprocal Rank Fusion (RRF)")

    vote_results = FusionStrategies.simple_voting(all_results)
    display_final_ranking(vote_results, "Voting Fusion")

    weights = [2.0, 1.0, 1.0, 1.0]
    weighted_results = FusionStrategies.weighted_fusion(all_results, weights)
    display_final_ranking(weighted_results, "Weighted Fusion")


if __name__ == "__main__":
    main()