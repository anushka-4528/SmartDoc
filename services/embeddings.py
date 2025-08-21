import os, google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "text-embedding-004"  # 768 dims

def get_embedding(text: str) -> list[float]:
    # safety truncate; generous enough
    text = text[:8000]
    out = genai.embed_content(model=EMBED_MODEL, content=text)
    return out["embedding"]

def get_embeddings(texts: list[str]) -> list[list[float]]:
    return [get_embedding(t) for t in texts]
