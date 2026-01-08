import os
import json
import pdfplumber
from langdetect import detect

RAW_DIR = "data/raw"
OUT_FILE = "data/processed/corpus.jsonl"

def clean_text(text):
    return " ".join(text.split())

def extract_pdf(path):
    txt = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt += page.extract_text() or ""
    return txt

def guess_site(fname):
    sites = ["Carthage", "Dougga", "El Jem", "Sbeitla", "Kerkouane", "Bulla Regia"]
    for s in sites:
        if s.lower() in fname.lower():
            return s
    return "Unknown"

def process_all():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for fname in os.listdir(RAW_DIR):
            if fname.lower().endswith(".pdf"):
                full = os.path.join(RAW_DIR, fname)
                print(f"Processing {fname}...")
                text = clean_text(extract_pdf(full))
                lang = detect(text)  # detect language

                doc = {
                    "title": fname.replace(".pdf", ""),
                    "source": fname,
                    "site": guess_site(fname),
                    "lang": lang,
                    "text": text
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Ingestion finished! Output saved to {OUT_FILE}")

if __name__ == "__main__":
    process_all()
