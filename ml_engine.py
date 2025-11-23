from transformers import pipeline as standard_pipeline
import torch
import torch.nn.functional as F

class MLEngine:
    def __init__(self):
        self.classifier = None
        self.generator = None

        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embedder = None
        self.generator_name = "Qwen/Qwen2.5-0.5B-Instruct"

        self.device = 0 if torch.cuda.is_available() else -1

    def load_model(self):
        print("-" * 50)

        print(f"Loading Embedder ({self.embedding_model_name})...")
        self.embedder = standard_pipeline(
            "feature-extraction",
            model=self.embedding_model_name,
            device=self.device
        )
        print("Embedder loaded.")

        # --- LOAD GENERATOR (Standard) ---
        print(f"Loading Generator ({self.generator_name})...")
        self.generator = standard_pipeline(
            "text-generation",
            model=self.generator_name,
            device=self.device,
            max_new_tokens=256,
            do_sample=False
        )

        print("-" * 50)

    def get_embedding(self, text: str):
        if not self.embedder:
            raise Exception("Embedder not loaded")

        output = self.embedder(text, truncation=True)[0]
        # Convert nested list → flat vector
        embedding = [float(x) for x in output]
        return embedding


    def predict(self, text: str, labels: list):
        """
        Predict using embedding similarity instead of zero-shot classifier.
        labels: list of strings (label descriptions)
        """
        if not self.embedder:
            raise Exception("Embedder (MPNet) not loaded")

        # ---- 1. Embed input text ----
        text_emb = self.embedder(text, truncation=True)[0]
        text_emb = torch.tensor(text_emb, dtype=torch.float32).mean(dim=0)  # pooled vector

        # ---- 2. Embed each label ----
        similarities = []
        for label in labels:
            label_emb = self.embedder(label, truncation=True)[0]
            label_emb = torch.tensor(label_emb, dtype=torch.float32).mean(dim=0)

            # ---- 3. Cosine similarity ----
            sim = F.cosine_similarity(text_emb, label_emb, dim=0).item()

            similarities.append({"label": label, "score": float(sim)})
        # ---- 4. Sort & return top 5 ----
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:5]

    def explain_prediction(self, text: str, label: str, score: float = None):
        if not self.generator:
            raise Exception("Generator model not loaded")

        score_info = f" The similarity score was {score:.3f}." if score else ""

        prompt = (
        f"Explain in one concise line why the transaction '{text}' matches the category '{label}'{score_info}, without restating the transaction description."
        )
        output = self.generator(
            prompt,
            max_new_tokens=30,
            do_sample=False,
            temperature=0.0,

        )

        result = output[0]["generated_text"]
        le=len(prompt)
        return f"This transaction aligns with the {label} category based on its meaning, supported by a similarity score of {score:3f}."+result[le:]


ml_instance = MLEngine()