from flask import Flask, render_template, request, jsonify
import spacy
from spacy import displacy
import fitz  # PyMuPDF for PDFs
from docx import Document
import os

app = Flask(__name__)
nlp = spacy.load("./output/model-best")

TEXT = ""

COLORS = {
    "PERSON": "lime",
    "LIT_WORK": "blue",
    "ART_WORK": "green",
    "ART_MOVEMENT": "orange",
    "ORG": "purple",
    "PLACE": "teal",
    "EVENT": "cyan",
    "GENRE": "pink",
    "CHARACTER": "brown",
    "QUOTE": "gold",
    "AWARD": "lime",
    "PERIOD": "magenta",
    "TECHNIQUE": "indigo",
    "MOVIE_TV": "violet"
}

# Extract entity labels from the model itself
MODEL_ENTITIES = sorted(nlp.get_pipe("ner").labels)

@app.route("/")
def index():
    # Ensure checkboxes always load from the model's entity labels
    entity_types = MODEL_ENTITIES  

    # Render empty text visualization
    doc = nlp(TEXT)
    options = {
        "ents": entity_types,
        "colors": {ent: COLORS.get(ent, "gray") for ent in entity_types}
    }
    html = displacy.render(doc, style="ent", options=options, page=False)

    return render_template("index.html", entity_types=entity_types, html=html)

@app.route("/filter", methods=["POST"])
def filter_entities():
    data = request.json
    selected_entities = data.get("selected_entities", [])
    user_text = data.get("text", "").strip()  # Get and clean user input text

    doc = nlp(user_text)

    # If no entity types are selected, don't highlight anything
    if not selected_entities:
        selected_entities = []

    # Render visualization **only** for selected entities
    options = {
        "ents": selected_entities,
        "colors": {ent: COLORS.get(ent, "gray") for ent in selected_entities}
    }
    html = displacy.render(doc, style="ent", options=options, page=False)

    return {"html": html}

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
        return jsonify({"error": "Invalid file format. Please upload a PDF or DOCX."}), 400

    #file_path = os.path.join("uploads", file.filename)
    #file.save(file_path)

    extracted_text = extract_text_from_file(file.filename)

    return jsonify({"text": extracted_text})

def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX file."""
    text = ""

    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.strip()

if __name__ == "__main__":
    app.run(debug=True)
