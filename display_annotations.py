import json
import spacy
from spacy.tokens import Doc, Span
from spacy import displacy

# Load a blank SpaCy model
nlp = spacy.blank("en")

# Load the annotations file
annotations_file = "annotations.jsonl"

# Read and parse annotations with error handling
try:
    with open(annotations_file, "r", encoding="utf-8") as f:
        annotations = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"Error: The file {annotations_file} was not found.")
    annotations = []
except json.JSONDecodeError as e:
    print(f"Error reading JSON: {e}")
    annotations = []

# Function to create a SpaCy Doc object with entity annotations
def create_doc(text, entities):
    doc = nlp(text)
    ents = []
    
    for entity in entities:
        if not isinstance(entity, (list, tuple)) or len(entity) != 3:
            print(f"Skipping malformed entity: {entity}")
            continue
        
        start, end, label = entity
        span = doc.char_span(start, end, label=label)  # More reliable for character offsets

        if span is not None:
            ents.append(span)
        else:
            print(f"Warning: Failed to create span for entity {entity}")

    doc.ents = ents
    return doc

# Process annotations
docs = [create_doc(ann["text"], ann.get("entities", [])) for ann in annotations if "text" in ann]

# Render the annotated text using displacy
if docs:
    displacy.serve(docs, style="ent", host="127.0.0.1")
else:
    print("No valid annotations to display.")
