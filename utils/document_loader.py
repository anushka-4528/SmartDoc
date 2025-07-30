from PyPDF2 import PdfReader
from docx import Document

def load_pdf(path):
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text

def load_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_txt(path):
    with open(path, 'r') as f:
        return f.read()
