# RAG Retriever 示例（Mock 文档召回）
# rag_explainer/retriever.py
def retrieve_docs(query):
    # 模拟的文档库
    documents = [
        "iPhone 15 is the latest Apple phone.",
        "MacBook Air has the M2 chip.",
        "Apple Watch is great for fitness tracking."
    ]
    # 简单示例：返回包含query关键词的文档
    result = [doc for doc in documents if any(word.lower() in doc.lower() for word in query.split())]
    # 如果没匹配上，返回前2条文档作为兜底
    return result if result else documents[:2]
