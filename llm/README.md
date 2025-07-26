# 🔮 llm-recsys-hybrid

A hybrid recommendation system that integrates traditional recommender models with the semantic understanding and knowledge of large language models (LLMs), enhanced with explainability via RAG (Retrieval-Augmented Generation) and support for OpenAI GPT interfaces.

---

## 📌 Key Features

- **Two-Tower Retriever**: Efficient user-item vector matching for candidate retrieval.
- **LLM-as-Ranker**: Use LLMs (e.g. Sentence Transformers or GPT) to rerank candidates with deep semantic understanding.
- **Prompt Engineering**: Transforms user behavior sequences into natural language prompts that LLMs can interpret.
- **RAG-Based Explanation**: Generates human-readable explanations for recommendations.
- **Knowledge Distillation**: Compress LLM knowledge into lightweight models to reduce inference latency.
- **OpenAI API Support**: Optional integration with GPT-3.5 / GPT-4 for enhanced reasoning and ranking.
- **MovieLens Compatible**: Includes sample MovieLens-style data for quick experiments.

---

## 📁 Project Structure
llm-recsys-hybrid/
├── data/ # Sample user and item data (incl. MovieLens)
├── retriever/ # Two-Tower candidate retriever
├── reranker/ # LLM-as-Ranker (local or OpenAI)
├── prompt_engineering/ # User behavior to prompt converter
├── rag_explainer/ # RAG-based explanation system
├── distillation/ # Knowledge distillation trainer
├── tests/ # Unit tests for key modules
├── config.py # Configs and paths
├── main.py # Entry point
├── requirements.txt
├── LICENSE
└── README.md

---

## ⚙️ Installation

```bash
git clone https://github.com/Susie-ma/llm-recsys-hybrid.git
cd llm-recsys-hybrid
pip install -r requirements.txt
🚀 Quick Start
1. Generate prompt from user behavior
from prompt_engineering.behavior_to_prompt import behavior_to_prompt

behavior = ["clicked iPhone 13", "searched MacBook", "browsed Apple Watch"]
prompt = behavior_to_prompt(behavior)
print(prompt)
2. Use SentenceTransformer-based reranker
from reranker.llm_ranker import LLMRanker

ranker = LLMRanker()
candidates = ["iPhone 15", "MacBook Air", "Samsung S23"]
results = ranker.rank(prompt, candidates)

for item, score in results:
    print(f"{item} - Score: {score}")
3. Use OpenAI GPT-based reranker (optional)
# Set your key before running
# export OPENAI_API_KEY=sk-xxx
from reranker.llm_openai_ranker import gpt_rank

results = gpt_rank(prompt, candidates)
4. Generate explanations using RAG
from rag_explainer.explainer import explain_recommendation

print(explain_recommendation(prompt))
🔑 Using OpenAI API
export OPENAI_API_KEY=your_openai_key

📊 Sample Data
data/sample_items.csv - Item catalog

data/sample_users.json - User behavior logs

data/movielens_sample.csv - MovieLens-style ratings

You can replace these with real data to test retrieval, reranking, and explanation.
🛠 TODO
 Integrate FAISS/BM25 retrieval

 Add LangChain support for RAG

 Online A/B testing logging module

 End-to-end API service (FastAPI/Flask)

📄 License
This project is licensed under the MIT License © Susie-ma 2025.


