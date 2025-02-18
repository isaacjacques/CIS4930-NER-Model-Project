import json
import os
import re
import wikipediaapi
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.cli.train import train
import random
from nltk.corpus import stopwords

nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Default filename for storing the entity dictionary
FILENAME = "entity_dict.json"
ANNOTATIONS_FILE = "annotations.jsonl"

def load_dictionary(filename=FILENAME):
    """Load the entity dictionary from a JSON file."""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}  # Return an empty dictionary if file doesn't exist

def save_dictionary(dictionary, filename=FILENAME):
    """Save the entity dictionary to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=4, ensure_ascii=False)

def add_item(entity_type, new_items, filename=FILENAME):
    """Add new items to a specific entity category, avoiding duplicates."""
    dictionary = load_dictionary(filename)

    if entity_type not in dictionary:
        dictionary[entity_type] = []  # Create category if it doesn't exist

    # Add unique items
    dictionary[entity_type].extend([item for item in new_items if item not in dictionary[entity_type]])

    save_dictionary(dictionary)  # Save changes
    print(f"Added {len(new_items)} item(s) to '{entity_type}'.")
    
def search_item(item_name, filename=FILENAME):
    """Search for an item and return its entity category if found."""
    dictionary = load_dictionary(filename)
    
    for category, items in dictionary.items():
        if item_name in items:
            return f"'{item_name}' found in category: {category}"
    
    return f"'{item_name}' not found in the dictionary."

def remove_item(entity_type, item_name, filename=FILENAME):
    """Remove an item from a specific entity category."""
    dictionary = load_dictionary(filename)
    
    if entity_type in dictionary and item_name in dictionary[entity_type]:
        dictionary[entity_type].remove(item_name)
        save_dictionary(dictionary)
        print(f"Removed '{item_name}' from '{entity_type}'.")
    else:
        print(f"'{item_name}' not found in '{entity_type}'.")

def display_dictionary(filename=FILENAME):
    """Display all entities and their categories."""
    dictionary = load_dictionary(filename)
    
    if not dictionary:
        print("The dictionary is empty.")
        return
    
    for category, items in dictionary.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")

def fetch_wikipedia_content(topic, output_dir="sources/wikipedia"):
    # Normalize topic for filename
    normalized_topic = normalize_topic(topic)
    filename = os.path.join(output_dir, f"{normalized_topic}.txt")
    
    # Check if file already exists
    if os.path.exists(filename):
        print(f"Skipping fetch: {filename} already exists.")
        return None
    
    user_agent = "WikiBot/1.0 (contact: ij24d@fsu.edu)"
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")
    page = wiki_wiki.page(topic)

    if not page.exists():
        print(f"The topic '{topic}' does not exist on Wikipedia.")
        return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extracting main text content
    content = page.text

    # preprocess the content
    content = normalize_text(content)
    content = format_paragraphs(content)
    
    # Tokenize into sentences
    sentences = sent_tokenize(content)

    # Writing to a file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(sentences))

    print(f"Content for '{topic}' has been saved to {filename}.")
    return page.fullurl

def sync_wikipedia(filename=FILENAME):
    """Search Wikipedia for topics in the entity dictionary and save content."""
    dictionary = load_dictionary(filename)
    
    for category, items in dictionary.items():
        print(f"Searching Wikipedia for {category}...")
        for item in items:
            print(f"Fetching: {item}")
            fetch_wikipedia_content(item)
  
def normalize_topic(topic):
    return re.sub(r'[<>:"/\\|?*]', '', topic.replace(" ", "_"))
          
def normalize_text(text):
    # Normalize Unicode characters
    text = text.replace("\u2019", "'").replace("\u2018", "'")  # Single quotes
    text = text.replace("\u201C", '"').replace("\u201D", '"')  # Double quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # Dashes
    text = text.replace("\u2026", "...")  # Ellipses

    # Convert to ASCII and normalize spacing
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
    text = text.replace(". . .", "...")  # Normalize ellipses
    return text.strip()

def format_paragraphs(text):
    # Merge broken lines into paragraphs
    return "\n".join(re.split(r'\n{2,}', text))

def annotate_data(input_dir="sources/wikipedia", output_file=ANNOTATIONS_FILE):
    dictionary = load_dictionary()
    annotations = []
    all_entities = []
    for category, entities in dictionary.items():
        all_entities.extend(entities)
    
    # Collect all .txt files in the input directory
    for entity_type, topics in dictionary.items():
        print(f"Working on {entity_type} entities")
        for topic in topics: 
            if topic != "William Shakespeare":
                 continue
            normalized_topic = normalize_topic(topic)
            filename = os.path.join(input_dir, f"{normalized_topic}.txt")
 
            # Check if file for topic exists
            if not os.path.exists(filename):
                print(f"Skipping topic: {filename} does not exists.")
                continue
           
            with open(filename, "r", encoding="utf-8") as f:
                print(f"Annotating {topic}")

                topic_parts = [word for word in topic.split() if word.lower() not in stop_words]

                pattern = r"\b(" + "|".join(map(re.escape, sorted(all_entities+topic_parts, key=len, reverse=True))) + r")\b"    

                sentences = f.readlines()
                for sentence in sentences:
                    entities = []
                    occupied_positions = set()
                    entity_counts = {}

                    # Use regex to find matches in the sentence
                    for match in re.finditer(pattern, sentence, re.IGNORECASE):
                        start, end = match.start(), match.end()
                        matched_text = match.group(0).lower()

                        # Ensure no overlapping entities
                        if any(pos in occupied_positions for pos in range(start, end)):
                            continue

                        entity_counts[matched_text] = entity_counts.get(matched_text, 0) + 1
                        # If any entity appears more than once, skip the sentence
                        if entity_counts[matched_text] > 1:
                            break

                        if matched_text in map(str.lower, [topic] + topic_parts):
                            entities.append([start, end, entity_type])
                        else:
                            for cat, items in dictionary.items():
                                if matched_text in map(str.lower, items):
                                    entities.append([start, end, cat])
                                    break
                            else:
                                print(f"Could not find category for matched text: {matched_text}")
                                continue  # Skip if not found

                        occupied_positions.update(range(start, end))  # Mark positions as used

                    if entities:
                        annotations.append({
                            "text": sentence.strip(),
                            "entities": entities
                        })

    # Save annotations in spaCy JSONL format
    with open(output_file, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")
    
    print(f"Annotations saved to {output_file}.")


def serialize_data(input_file=ANNOTATIONS_FILE, split_ratio=0.8):
    # Load a blank English model
    nlp = spacy.blank("en")
    # Create lists to store training and development examples
    docs = []
    # Load annotated data
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            data = json.loads(line.strip())
            text = data["text"]
            entities = data["entities"]

            # Create a spaCy doc object
            doc = nlp.make_doc(text)

            # Add annotations
            ents = []
            seen_tokens = set()
            for start, end, label in entities:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")

                if span is None:
                    print(f"Line {line_num}: Invalid span '{text[start:end]}' [{start}:{end}] in '{text}'")
                    continue  # Skip invalid spans

                # Prevent overlapping entities
                if any(token.i in seen_tokens for token in span):
                    print(f"Line {line_num}: Overlapping entity '{span.text}' [{start}:{end}] in '{text}'")
                    input
                    continue  

                ents.append(span)
                seen_tokens.update(token.i for token in span)
                
            # Apply entity annotations
            doc.ents = ents
            docs.append(doc)

    # Shuffle data and split into train and dev sets
    random.shuffle(docs)
    split_point = int(len(docs) * split_ratio)
    train_docs = docs[:split_point]
    dev_docs = docs[split_point:]

    # Convert to spaCy's binary format
    train_doc_bin = DocBin(docs=train_docs)
    dev_doc_bin = DocBin(docs=dev_docs)

    # Save the datasets
    train_doc_bin.to_disk("train_data.spacy")
    dev_doc_bin.to_disk("dev_data.spacy")

    print(f"Training data saved as train_data.spacy ({len(train_docs)} examples)")
    print(f"Development data saved as dev_data.spacy ({len(dev_docs)} examples)")



if __name__ == "__main__":
    while True:
        print("\nEntity Dictionary Manager")
        print("1. Add item")
        print("2. Search item")
        print("3. Display dictionary")
        print("4. Sync data")
        print("5. Annotate data")
        print("6. Serialize data")
        print("7. Train Model")
        print("8. Exit")

        choice = input("Select an option: ")

        if choice == "1":
            entity_type = input("Enter entity category: ")
            new_items = input("Enter new items (comma-separated): ").split(",")
            add_item(entity_type.strip(), [item.strip() for item in new_items])

        elif choice == "2":
            item_name = input("Enter item name to search: ").strip()
            print(search_item(item_name))

        elif choice == "3":
            display_dictionary()

        elif choice == "4":
            sync_wikipedia()

        elif choice == "5":
            annotate_data()

        elif choice == "6":
            serialize_data()

        elif choice == "7":
            train("config.cfg", output_path="./output")

        elif choice == "8":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")
