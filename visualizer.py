from flask import Flask, render_template, request, jsonify
import spacy
from spacy import displacy
import fitz  # PyMuPDF for PDFs
from docx import Document
from docx.shared import RGBColor
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

ENTITY_COLORS = {
    "PERSON": RGBColor(50, 205, 50),   # Lime
    "LIT_WORK": RGBColor(0, 0, 255),   # Blue
    "ART_WORK": RGBColor(0, 128, 0),   # Green
    "ART_MOVEMENT": RGBColor(255, 165, 0),  # Orange
    "ORG": RGBColor(128, 0, 128),  # Purple
    "PLACE": RGBColor(0, 128, 128),  # Teal
    "EVENT": RGBColor(0, 255, 255),  # Cyan
    "GENRE": RGBColor(255, 20, 147),  # Pink
    "CHARACTER": RGBColor(165, 42, 42),  # Brown
    "QUOTE": RGBColor(255, 215, 0),  # Gold
    "AWARD": RGBColor(50, 205, 50),  # Lime
    "PERIOD": RGBColor(255, 0, 255),  # Magenta
    "TECHNIQUE": RGBColor(75, 0, 130),  # Indigo
    "MOVIE_TV": RGBColor(148, 0, 211)  # Violet
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

@app.route("/save", methods=["POST"])
def save_results():
    data = request.json
    text = data.get("text", "").strip()
    selected_entities = data.get("selected_entities", [])

    if not text:
        return jsonify({"error": "No content to save"}), 400

    doc = Document()
    para = doc.add_paragraph()

    # Process text with NLP model
    nlp_text = nlp(text)

    # Apply entity color formatting
    last_index = 0
    for ent in nlp_text.ents:
        if ent.label_ in selected_entities:
            # Add normal text before entity
            if ent.start_char > last_index:
                para.add_run(text[last_index:ent.start_char])

            # Add colored entity text
            entity_run = para.add_run(ent.text)
            entity_run.bold = True
            entity_run.font.color.rgb = ENTITY_COLORS.get(ent.label_, RGBColor(128, 128, 128))  # Default gray

            last_index = ent.end_char

    # Add remaining text after last entity
    if last_index < len(text):
        para.add_run(text[last_index:])

    # Save as .docx file
    file_path = "labeled_text.docx"
    doc.save(file_path)

    return jsonify({"success": True, "file_path": file_path})

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
