# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid, io, csv
from dotenv import load_dotenv

# --- File parsing / OCR ---
import fitz  # PyMuPDF
import docx
import openpyxl
import pytesseract
from PIL import Image

# --- Gemini (gen + embeddings) ---
import google.generativeai as genai

# --- Vector DB (Pinecone) ---
from pinecone import Pinecone, ServerlessSpec

# --- Your prompt helpers ---
from prompts import build_prompt, build_context, parse_llm_json

# =========================================================
# Config
# =========================================================
load_dotenv()

GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY")
MODEL_NAME         = os.getenv("MODEL_NAME", "gemini-1.5-flash")
TEMPERATURE        = float(os.getenv("TEMPERATURE", "0.2"))
MAX_CONTEXT_CHARS  = int(os.getenv("MAX_CHARS", "12000"))

# Vector config
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "smartdocq")
PINECONE_CLOUD     = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION    = os.getenv("PINECONE_REGION", "us-east-1")

TOP_K              = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "200"))

# Gemini setup
genai.configure(api_key=GOOGLE_API_KEY)
EMBED_MODEL = "text-embedding-004"  # 768 dims

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory stores (OK for milestone/demo)
DOC_TEXTS = {}        # doc_id -> extracted text
DOC_META  = {}        # doc_id -> metadata
SESSION_MEMORY = {}   # session_id -> [ {q, a} ]

# =========================================================
# Vector DB helpers (Pinecone)
# =========================================================
pc = None
pindex = None
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Create index if missing (handy for dev)
    try:
        existing = [i["name"] for i in pc.list_indexes()]
        if PINECONE_INDEX not in existing:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=768,           # text-embedding-004
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
        pindex = pc.Index(PINECONE_INDEX)
    except Exception as _e:
        # let health endpoint surface vector_ready = false
        pindex = None

def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        if j == n: break
        i = j - overlap
    return chunks

def embed_text(text: str) -> list[float]:
    # safety truncate; 8k is plenty for good embeddings
    text = text[:8000]
    out = genai.embed_content(model=EMBED_MODEL, content=text)
    return out["embedding"]

def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_text(t) for t in texts]

def upsert_chunks_to_pinecone(doc_id: str, chunks: list[str], vectors: list[list[float]]):
    if not pindex:
        raise RuntimeError("Pinecone not configured. Set PINECONE_API_KEY / PINECONE_INDEX.")
    payload = []
    for i, (txt, vec) in enumerate(zip(chunks, vectors)):
        payload.append({
            "id": f"{doc_id}:{i}",
            "values": vec,
            "metadata": {"doc_id": doc_id, "seq": i, "snippet": txt[:200]}
        })
    # Batch upsert (good default batch size)
    B = 100
    for b in range(0, len(payload), B):
        pindex.upsert(vectors=payload[b:b+B])

def query_topk_from_pinecone(query_vec: list[float], top_k: int = TOP_K, filter_doc: str | None = None):
    if not pindex:
        raise RuntimeError("Pinecone not configured. Set PINECONE_API_KEY / PINECONE_INDEX.")
    flt = {"doc_id": {"$eq": filter_doc}} if filter_doc else None
    res = pindex.query(vector=query_vec, top_k=top_k, include_metadata=True, filter=flt)
    matches = res.get("matches", []) or []
    return [{
        "id": m["id"],
        "score": m["score"],
        "doc_id": m["metadata"].get("doc_id"),
        "seq": m["metadata"].get("seq"),
        "snippet": m["metadata"].get("snippet", "")
    } for m in matches]

# =========================================================
# Extraction helpers
# =========================================================
def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        return extract_pdf(filepath)
    elif ext == '.txt':
        return extract_txt(filepath)
    elif ext == '.docx':
        return extract_docx(filepath)
    elif ext == '.csv':
        return extract_csv(filepath)
    elif ext == '.xlsx':
        return extract_xlsx(filepath)
    elif ext in ('.jpg', '.jpeg', '.png'):
        return extract_image(filepath)
    else:
        raise ValueError("Unsupported file type")

def extract_pdf(filepath):
    text = ""
    pages = 0
    with fitz.open(filepath) as doc:
        pages = len(doc)
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            else:
                # OCR fallback for scanned page
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img)
                text += ocr_text
    return text, pages

def extract_txt(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read(), None

def extract_docx(filepath):
    d = docx.Document(filepath)
    text = "\n".join([p.text for p in d.paragraphs])
    return text, None

def extract_csv(filepath):
    lines = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(", ".join(row))
    return "\n".join(lines), None

def extract_xlsx(filepath):
    wb = openpyxl.load_workbook(filepath, data_only=True)
    text = ""
    for sheet in wb:
        for row in sheet.iter_rows(values_only=True):
            text += ", ".join([str(cell) for cell in row if cell is not None]) + "\n"
    return text, None

def extract_image(filepath):
    img = Image.open(filepath)
    text = pytesseract.image_to_string(img)
    return text, None

# =========================================================
# Routes
# =========================================================
@app.post('/upload')
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = os.path.basename(file.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        extracted, pages = extract_text(filepath)
        if not extracted.strip():
            return jsonify({"message": "File uploaded, but no text extracted.", "text": ""}), 200

        doc_id = str(uuid.uuid4())
        DOC_TEXTS[doc_id] = extracted
        size = os.path.getsize(filepath)
        DOC_META[doc_id] = {"filename": filename, "pages": pages, "size_bytes": size}

        return jsonify({
            "message": "File uploaded and text extracted successfully.",
            "doc_id": doc_id,
            "meta": DOC_META[doc_id],
            "text": extracted
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500

@app.post('/embed_doc')
def embed_doc():
    """Chunk + embed + upsert one uploaded document into Pinecone."""
    data = request.get_json() or {}
    doc_id = data.get("doc_id")
    if not doc_id or doc_id not in DOC_TEXTS:
        return jsonify({"error": "Invalid or missing doc_id"}), 400
    try:
        text = DOC_TEXTS[doc_id]
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        vectors = embed_texts(chunks)
        upsert_chunks_to_pinecone(doc_id, chunks, vectors)
        return jsonify({"message": "Embeddings stored", "doc_id": doc_id, "chunks": len(chunks)}), 200
    except Exception as e:
        return jsonify({"error": f"Embedding/upsert failed: {str(e)}"}), 500

@app.post('/retrieve_top5')
def retrieve_top5():
    """
    Body: { "query": "text...", "doc_id": "<optional>" }
    Returns: top-5 most similar chunks (cosine) from vector DB with chunk ids.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    doc_id = data.get("doc_id")  # optional filter

    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    try:
        q_vec = embed_text(query)
        results = query_topk_from_pinecone(q_vec, top_k=5, filter_doc=doc_id)
        return jsonify({"matches": results}), 200
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

@app.post('/ask')
def ask_question():
    """RAG: embed question -> retrieve top-K chunks -> build prompt -> Gemini -> answer (+memory & sources)."""
    data = request.get_json() or {}
    question     = data.get('question', '').strip()
    doc_id       = data.get('doc_id')              # optional filter
    session_id   = data.get('session_id', 'default')
    style        = data.get('style', 'concise')
    citation_mode= bool(data.get('citation_mode', True))
    model_name   = data.get('model', MODEL_NAME)
    temperature  = float(data.get('temperature', TEMPERATURE))
    top_k        = int(data.get('top_k', TOP_K))

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400
    if not DOC_TEXTS:
        return jsonify({"error": "No document uploaded yet."}), 400

    try:
        # Session memory
        history = SESSION_MEMORY.setdefault(session_id, [])
        history_text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])

        # Retrieval (if Pinecone configured)
        candidate_chunks = None
        hits = []
        if pindex:
            q_vec = embed_text(question)
            hits = query_topk_from_pinecone(q_vec, top_k=top_k, filter_doc=doc_id)
            if hits:
                candidate_chunks = [h["snippet"] for h in hits]

        # Fallback to raw doc text if retrieval not available/empty
        if candidate_chunks:
            base_context_text = history_text
        else:
            ctx_text = DOC_TEXTS.get(doc_id) if (doc_id and doc_id in DOC_TEXTS) else next(reversed(DOC_TEXTS.values()), "")
            base_context_text = (ctx_text + "\n" + history_text).strip()

        # Build prompt
        context = build_context(base_context_text, candidate_chunks=candidate_chunks, max_chars=MAX_CONTEXT_CHARS)
        prompt  = build_prompt(context=context, question=question, style=style, citation_mode=citation_mode)

        # Generate
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        raw_text = getattr(response, "text", "") or ""
        parsed = parse_llm_json(raw_text)

        # Save memory
        history.append({"q": question, "a": parsed.get("answer", "")})

        # Attach sources if we had retrieval hits
        if hits:
            parsed["sources"] = hits
        parsed["doc_id"] = doc_id
        parsed["session_id"] = session_id

        return jsonify(parsed), 200
    except Exception as e:
        return jsonify({"error": f"Failed to get response: {str(e)}"}), 500

@app.get("/docs")
def list_docs():
    return jsonify({"docs": [{"doc_id": did, **meta} for did, meta in DOC_META.items()]}), 200

@app.post("/reset")
def reset_state():
    DOC_TEXTS.clear(); DOC_META.clear(); SESSION_MEMORY.clear()
    return jsonify({"message": "All state cleared"}), 200

@app.get("/healthz")
def healthz():
    status = "ok"
    vec = bool(pindex)
    return jsonify({"status": status, "model": MODEL_NAME, "vector_ready": vec}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
