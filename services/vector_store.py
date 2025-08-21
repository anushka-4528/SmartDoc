import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "smartdocq")

def _index():
    return pc.Index(INDEX_NAME)

def upsert_chunks(doc_id: str, chunks: list[str], vectors: list[list[float]]):
    payload = []
    for i, (txt, vec) in enumerate(zip(chunks, vectors)):
        payload.append({
            "id": f"{doc_id}:{i}",               # <-- chunk id format
            "values": vec,
            "metadata": {
                "doc_id": doc_id,
                "seq": i,
                "snippet": txt[:200]             # small preview
            }
        })
    # batch upsert
    B = 100
    for b in range(0, len(payload), B):
        _index().upsert(vectors=payload[b:b+B])

def query_top_k(vector: list[float], top_k: int = 5, filter_doc: str | None = None):
    flt = {"doc_id": {"$eq": filter_doc}} if filter_doc else None
    res = _index().query(vector=vector, top_k=top_k, include_metadata=True, filter=flt)
    matches = res.get("matches", []) or []
    # Pinecone already uses cosine similarity (set at index creation)
    # Higher score = more similar for cosine in Pinecone
    return [{
        "chunk_id": m["id"],                  # e.g., "<doc_id>:<seq>"
        "score": m["score"],
        "doc_id": m["metadata"].get("doc_id"),
        "seq": m["metadata"].get("seq"),
        "snippet": m["metadata"].get("snippet", "")
    } for m in matches]
