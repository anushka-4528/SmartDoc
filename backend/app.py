from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import docx
import csv
import openpyxl
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store extracted text
extracted_text_store = ""

# Helper to extract text based on file extension
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
    else:
        raise ValueError("Unsupported file type")

def extract_pdf(filepath):
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def extract_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_csv(filepath):
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(", ".join(row))
    return "\n".join(lines)

def extract_xlsx(filepath):
    wb = openpyxl.load_workbook(filepath)
    text = ""
    for sheet in wb:
        for row in sheet.iter_rows(values_only=True):
            text += ", ".join([str(cell) for cell in row if cell is not None]) + "\n"
    return text

@app.route('/upload', methods=['POST'])
def upload_file():
    global extracted_text_store

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        extracted_text = extract_text(filepath)

        if not extracted_text.strip():
            return jsonify({
                "message": "File uploaded, but no text was extracted.",
                "text": ""
            }), 200

        extracted_text_store = extracted_text
        return jsonify({
            "message": "File uploaded and text extracted successfully.",
            "text": extracted_text
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global extracted_text_store
    data = request.get_json()
    question = data.get('question', '')

    if not extracted_text_store:
        return jsonify({"error": "No document uploaded yet."}), 400

    try:
        prompt = f"Answer this question based on the document:\n\n{extracted_text_store}\n\nQuestion: {question}"
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return jsonify({"answer": response.text}), 200
    except Exception as e:
        print("Gemini API Error:", str(e))
        return jsonify({"error": f"Failed to get response: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5001)
