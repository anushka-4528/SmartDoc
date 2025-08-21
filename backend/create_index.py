# backend/create_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # loads backend/.env when you run from backend/

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX", "smartdocq")
cloud = os.getenv("PINECONE_CLOUD", "aws")
region = os.getenv("PINECONE_REGION", "us-east-1")

if not api_key:
    raise SystemExit("PINECONE_API_KEY is not set. Add it to backend/.env")

pc = Pinecone(api_key=api_key)

# list existing indexes
existing = [i["name"] for i in pc.list_indexes()]

if index_name in existing:
    print(f"Index already exists: {index_name}")
else:
    # Gemini text-embedding-004 -> 768 dimensions
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    print(f"Created index: {index_name} in {cloud}/{region}")
