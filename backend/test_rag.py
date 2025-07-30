query = "What is the document talking about?"
query_embedding = get_embedding(query)
results = collection.query(query_embeddings=[query_embedding], n_results=5)

context = "\n".join(results["documents"][0])
prompt = f"""Use the following context to answer the question:
{context}

Question: {query}
"""

response = model.generate_content(prompt)
print("\n--- RAG Response ---\n")
print(response.text)
