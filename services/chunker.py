def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        if j == n: break
        i = j - overlap
    return chunks
