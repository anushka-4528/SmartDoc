import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = genai.GenerativeModel("embedding-001")

def get_embedding(text):
    return embedding_model.embed_content(text, task_type="RETRIEVAL_DOCUMENT")["embedding"]
