# rag_explainer/rag_demo.py
from transformers import pipeline

def generate_explanation(prompt, docs):
    context = " ".join(docs)
    generator = pipeline("text-generation", model="gpt2")
    input_text = f"{prompt}\nContext: {context}\nExplanation:"
    outputs = generator(input_text, max_length=100, do_sample=False)
    return outputs[0]['generated_text']

if __name__ == "__main__":
    user_prompt = "User clicked iPhone 13 and searched MacBook."
    retrieved_docs = [
        "iPhone 15 is the latest Apple phone.",
        "MacBook Air has the M2 chip."
    ]
    explanation = generate_explanation(user_prompt, retrieved_docs)
    print("Generated Explanation:", explanation)

