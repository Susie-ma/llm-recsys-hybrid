# RAG 示例流程：推荐解释生成
from transformers import pipeline

def generate_explanation(prompt, top_k_docs):
    context = " ".join(top_k_docs)
    generator = pipeline("text-generation", model="gpt2")
    input_text = f"{prompt}\nContext: {context}\nExplanation:"
    response = generator(input_text, max_length=100, do_sample=False)[0]['generated_text']
    return response

if __name__ == "__main__":
    user_prompt = "User showed interest in iPhones and MacBooks."
    docs = ["iPhone 15 is the latest Apple phone.", "MacBook Air has the M2 chip."]
    explanation = generate_explanation(user_prompt, docs)
    print("🔍 Explanation:", explanation)
