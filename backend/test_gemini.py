import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env file (optional but recommended)
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

prompt = "Explain what Retrieval-Augmented Generation (RAG) is in simple terms."
response = model.generate_content(prompt)

print(response.text)
