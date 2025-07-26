# LLM-as-Ranker reranker using sentence-transformers or OpenAI
from sentence_transformers import SentenceTransformer, util

class LLMRanker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def rank(self, prompt, candidates):
        prompt_emb = self.model.encode(prompt, convert_to_tensor=True)
        candidate_embs = self.model.encode(candidates, convert_to_tensor=True)
        scores = util.cos_sim(prompt_emb, candidate_embs)[0]
        ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked

if __name__ == "__main__":
    ranker = LLMRanker()
    prompt = "User recently clicked iPhone 13 and searched MacBook."
    candidates = ["iPhone 15", "MacBook Air", "Samsung S23"]
    ranked = ranker.rank(prompt, candidates)
    for item, score in ranked:
        print(f"{item}: {score:.4f}")
