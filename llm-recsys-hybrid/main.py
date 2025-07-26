# Entry point for running the hybrid recommendation system
# demo_pipeline.py

from prompt_engineering.behavior_to_prompt import behavior_to_prompt
from retriever.two_tower import TwoTowerRetriever
from reranker.llm_ranker import LLMRanker
from rag_explainer.retriever import retrieve_docs
from rag_explainer.rag_demo import generate_explanation

def run_pipeline(user_behavior, candidate_items):
    print("1. Convert user behavior to prompt...")
    prompt = behavior_to_prompt(user_behavior)
    print("Prompt:", prompt, "\n")

    print("2. Recall candidates using Two-Tower retriever...")
    retriever = TwoTowerRetriever()
    recalled = retriever.recall(user_behavior[0], candidate_items, top_k=5)
    candidates = [item for item, score in recalled]
    print("Recalled candidates:", candidates, "\n")

    print("3. Rank candidates with LLM ranker...")
    ranker = LLMRanker()
    ranked = ranker.rank(prompt, candidates)
    print("Ranked candidates and scores:")
    for item, score in ranked:
        print(f" - {item}: {score:.4f}")
    print()

    print("4. Retrieve docs for explanation...")
    docs = retrieve_docs(prompt)
    print("Retrieved docs:", docs, "\n")

    print("5. Generate explanation with RAG...")
    explanation = generate_explanation(prompt, docs)
    print("Generated explanation:\n", explanation)


if __name__ == "__main__":
    user_behavior = ["clicked iPhone 13", "searched MacBook", "browsed Apple Watch"]
    candidate_items = ["iPhone 15", "MacBook Air", "Apple Watch", "Samsung S23", "Dell Laptop"]
    run_pipeline(user_behavior, candidate_items)
