# backend/prompts.py
from __future__ import annotations
import json
from typing import List, Optional, Dict, Any


def build_context(
    full_text: str,
    candidate_chunks: Optional[List[str]] = None,
    max_chars: int = 12000
) -> str:
    """
    Builds the final context string used in the RAG prompt.
    - If candidate_chunks (retrieved top-K) are provided, we prioritize them.
    - We still include a trimmed full_text (history or fallback) to help disambiguate.
    """
    full = (full_text or "")[:max_chars]
    parts = []
    if candidate_chunks:
        # Number chunks so the model can cite them as [C1], [C2], ...
        numbered = [f"[C{i+1}] {c}" for i, c in enumerate(candidate_chunks)]
        parts.append("Retrieved chunks:\n" + "\n\n".join(numbered))
    if full:
        parts.append("Additional context:\n" + full)
    return "\n\n".join(parts).strip() or "No context."


def build_prompt(
    context: str,
    question: str,
    style: str = "concise",
    citation_mode: bool = True
) -> str:
    """
    Instructs the model to output strict JSON:
      { "answer": "...", "citations": ["C1", "C3"] }
    Citations refer to the [C#] tags we inject for retrieved chunks.
    """
    return f"""
You are a careful RAG assistant. Follow these rules:

1) Use ONLY the provided context to answer. If the answer is not present, say "I don't see that in the document."
2) If you cite, reference the chunk IDs like "C1", "C2" (from the [C#] tags).
3) Be {style}. No extra preamble, no disclaimers.
4) Output STRICT JSON with keys:
   - "answer": string
   - "citations": array of strings (e.g., ["C2", "C3"]). If none, use [].

Context:
{context}

User question:
{question}

Now return ONLY a JSON object with fields "answer" and "citations".
""".strip()


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """
    Attempts to parse the model response as JSON.
    If the model returned extra text, try to locate a JSON object; otherwise fall back.
    """
    raw = (raw_text or "").strip()

    # Fast path
    try:
        obj = json.loads(raw)
        # normalize fields
        ans = obj.get("answer", "")
        cits = obj.get("citations", [])
        if not isinstance(cits, list):
            cits = []
        return {"answer": ans, "citations": cits}
    except Exception:
        pass

    # Fallback: try to extract the first JSON object substring
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start:end+1])
            ans = obj.get("answer", "")
            cits = obj.get("citations", [])
            if not isinstance(cits, list):
                cits = []
            return {"answer": ans, "citations": cits}
        except Exception:
            pass

    # Last resort: return plain text
    return {"answer": raw, "citations": []}
