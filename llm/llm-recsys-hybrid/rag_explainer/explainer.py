# 调用 RAG Demo 接口
from rag_explainer.rag_demo import generate_explanation
from rag_explainer.retriever import retrieve_docs

def explain_recommendation(user_prompt):
    docs = retrieve_docs(user_prompt)
    return generate_explanation(user_prompt, docs)

if __name__ == "__main__":
    prompt = "User clicked iPhone and searched MacBook"
    print(explain_recommendation(prompt))
