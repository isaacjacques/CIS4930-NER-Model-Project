from flask import Flask, render_template, request
import spacy
from spacy import displacy

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

if __name__ == "__main__":
    app.run(debug=True)
