# 使用 OpenAI GPT 接口进行语义排序
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_rank(prompt, candidates):
    ranked = []
    for item in candidates:
        query = f"Prompt: {prompt}\nCandidate: {item}\nRelevance score (0-10):"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}]
        )
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
        except:
            score = 0.0
        ranked.append((item, score))
    return sorted(ranked, key=lambda x: -x[1])
