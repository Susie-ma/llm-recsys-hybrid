import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_rank(prompt, candidates):
    ranked = []
    for item in candidates:
        query = f"Prompt: {prompt}\\nCandidate: {item}\\nRate relevance 0-10:"
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
    ranked.sort(key=lambda x: -x[1])
    return ranked

if __name__ == "__main__":
    prompt = "User recently clicked iPhone 13 and searched MacBook."
    candidates = ["iPhone 15", "MacBook Air", "Samsung S23"]
    result = gpt_rank(prompt, candidates)
    for item, score in result:
        print(f"{item}: {score:.2f}")

