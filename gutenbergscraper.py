import os
import json
import requests
import re
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def normalize_text(text):
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201C", '"').replace("\u201D", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2026", "...")
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(". . .", "...")
    return text.strip()

def format_paragraphs(text):
    return "\n".join(re.split(r'\n{2,}', text))

def preprocess_text(text):
    # Use regex to find the book title and trim content accordingly
    start_marker_pattern = re.compile(r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK (.*?) \*\*\*', re.IGNORECASE)
    start_legal_pattern = re.compile(r'\*END\*', re.IGNORECASE)
    end_marker_pattern = re.compile(r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK (.*?) \*\*\*', re.IGNORECASE)

    start_match = None

    # Search for the last occurrence of *END*
    matches = list(start_legal_pattern.finditer(text))
    if matches:
        start_match = matches[-1]  # Use the last occurrence

    # If no *END* is found, fall back to the first occurrence of the standard start marker
    if start_match is None:
        start_match = start_marker_pattern.search(text)

    # Trim text based on found markers
    if start_match:
        text = text[start_match.end():].strip()

    end_match = end_marker_pattern.search(text)
    if end_match:
        text = text[:end_match.start()].strip()
    else:
        print("text footer not found!")

    text = normalize_text(text)
    text = format_paragraphs(text)
    sentences = sent_tokenize(text)
    return "\n".join(sentences)

def load_downloaded_books(json_path="books.json"):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def save_downloaded_books(data, json_path="books.json"):
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def extract_title(text):
    lines = text.split("\n")[:20]
    for line in lines:
        if line.startswith("Title: "):
            return line.replace("Title: ", "").strip()
    return "Unknown Title"

def download_gutenberg_books(start_id=1, end_id=100):
    base_url = "https://www.gutenberg.org/ebooks/{}.txt.utf-8"
    save_dir = "sources/gutenberg"
    output_dir = "trainingdata/preprocess"
    books_file = "books.json"
    json_path = os.path.join(save_dir, books_file)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_books = load_downloaded_books(json_path)
    
    for book_id in range(start_id, end_id + 1):
        book_url = base_url.format(book_id)
        file_path = os.path.join(save_dir, f"{book_id}.txt")
     
        if str(book_id) in downloaded_books:
            print(f"Skipping {book_id}: Already recorded in JSON")
            continue

        try:
            response = requests.get(book_url, timeout=10)
            response.raise_for_status()
            
            raw_text = response.text.replace("\r\n", "\n")
            title = extract_title(raw_text)
            processed_text = preprocess_text(raw_text)
            
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(processed_text)
            
            downloaded_books[str(book_id)] = title
            save_downloaded_books(downloaded_books, json_path)
            
            print(f"Downloaded and Processed: {book_id}.txt | Title: {title}")
        except requests.exceptions.RequestException as e:
            print(f"Skipping {book_id}: {e}")

if __name__ == "__main__":
    download_gutenberg_books(11,2000)
