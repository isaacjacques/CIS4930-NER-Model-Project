import spacy
import os
from spacy import displacy

# Load a small English model
nlp = spacy.load("./output/model-best") #spacy.load("en_core_web_sm")

case1 = "_testcase1.txt"
text = None

with open(case1, "r", encoding="utf-8") as f:
    text = f.read()
    doc = nlp(text)

# Render output
if text is not None:
    displacy.serve(doc, style="ent", host="127.0.0.1")
