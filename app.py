import os
import csv
import string
import random
from datetime import datetime
from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup BLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Setup Translator
translator = Translator()

# Load vibes list
def load_vibes(filepath):
    vibes = []
    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'vibes' in row:
                    vibes.append(row['vibes'])
    except Exception as e:
        print(f"Error membaca CSV vibes: {e}")
    return vibes

vibes_list = load_vibes("vibes_caption.csv")

# Setup semantic similarity model
similarity_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vibes_embeddings = similarity_model.encode(vibes_list, convert_to_tensor=True)

used_vibes_map = {}  # Hindari duplikat vibes per gambar

def preprocess(text):
    stopwords = set(["yang", "di", "ke", "dengan", "dan", "atau", "dari", "untuk", "pada", "adalah", "itu", "ini", "sebuah", "seorang"])
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return [word for word in text.split() if word not in stopwords]

def match_vibes_semantic(translated_caption, filename):
    query_vec = similarity_model.encode(translated_caption, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_vec, vibes_embeddings)[0]

    ranked_indices = similarity_scores.argsort(descending=True)
    used = used_vibes_map.get(filename, set())

    for idx in ranked_indices:
        vibe = vibes_list[idx]
        if vibe not in used:
            used.add(vibe)
            used_vibes_map[filename] = used
            return vibe
    return random.choice(vibes_list)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def translate_caption(text):
    try:
        translated = translator.translate(text, src='en', dest='id')
        return translated.text
    except:
        return text

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        files = request.files.getlist("images")
        for file in files:
            if file and file.filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                caption_en = generate_caption(filepath)
                translated = translate_caption(caption_en)
                final_caption = match_vibes_semantic(translated, file.filename)

                results.append({
                    "image": filepath,
                    "caption_en": caption_en,
                    "translated": translated,
                    "final_caption": final_caption,
                    "filename": file.filename
                })

    return render_template("index.html", results=results)

@app.route("/regenerate", methods=["POST"])
def regenerate():
    filename = request.form.get("filename")
    caption_en = request.form.get("caption_en")
    translated = request.form.get("translated")

    new_caption = match_vibes_semantic(translated, filename)

    return {
        "final_caption": new_caption
    }

if __name__ == "__main__":
    app.run(debug=True)
