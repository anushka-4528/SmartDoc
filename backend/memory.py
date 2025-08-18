# backend/memory.py
from collections import deque

class SessionMemory:
    def __init__(self, max_turns: int = 5):
        # Store last N (question, answer) pairs
        self.history = deque(maxlen=max_turns)

    def add_turn(self, question: str, answer: str):
        """Save one interaction."""
        self.history.append({"q": question, "a": answer})

    def get_history(self) -> str:
        """Return formatted history for prompt injection."""
        if not self.history:
            return ""
        formatted = []
        for i, turn in enumerate(self.history, start=1):
            formatted.append(f"Q{i}: {turn['q']}\nA{i}: {turn['a']}")
        return "\n".join(formatted)

    def clear(self):
        self.history.clear()
