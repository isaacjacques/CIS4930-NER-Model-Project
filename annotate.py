import spacy
import json
import os

nlp = spacy.load("en_core_web_sm")

input_dir = "trainingdata/preprocess"
output_dir = "trainingdata/annotated"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace(".txt", ".jsonl"))

        count_lines = 0
        count_lines_skipped = 0
        count_entities = 0

        with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    count_lines += 1
                    doc = nlp(line)
                    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

                    if entities:  # Only write lines with entities
                        count_entities += len(entities)  # FIXED: Count entities properly
                        annotation = {"text": line, "annotations": entities}
                        out_f.write(json.dumps(annotation) + "\n")
                    else:
                        count_lines_skipped += 1

        print(f"Annotated {input_file} | {count_entities} entities found | {count_lines} lines processed | {count_lines_skipped} lines skipped")
        print(f"\tAnnotated file saved to {output_file}")