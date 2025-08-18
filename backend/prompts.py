# prompts.py

def build_prompt(context, question, style="concise", citation_mode=True):
    return f"""
You are a helpful assistant. Use the provided context and conversation history to answer the user’s question.

Context:
{context}

Question: {question}
Answer style: {style}
Citations required: {citation_mode}
Provide the answer in JSON with keys: "answer", "citations" (if any).
"""

def build_context(full_text, candidate_chunks=None, max_chars=12000):
    text = (full_text or "")[:max_chars]
    return f"Document context:\n{text}\n"

def parse_llm_json(raw_text):
    import json
    try:
        return json.loads(raw_text)
    except Exception:
        return {"answer": raw_text.strip(), "citations": []}

class SessionMemory:
    def __init__(self, max_turns=5):
        self.turns = []
        self.max_turns = max_turns

    def add_turn(self, question, answer):
        self.turns.append({"q": question, "a": answer})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def build_context(self):
        if not self.turns:
            return ""
        context = "\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in self.turns])
        return f"Conversation history:\n{context}\n"
