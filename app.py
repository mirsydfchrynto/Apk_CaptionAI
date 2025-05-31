import os
import csv
import string
import random
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
import torch
from sentence_transformers import SentenceTransformer, util

# === Flask Setup ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === BLIP Captioning Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# === Google Translate Setup ===
translator = Translator()

# === Load Vibes from CSV ===
def load_vibes(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [row['vibes'] for row in reader if 'vibes' in row and row['vibes'].strip()]
    except Exception as e:
        print(f"[ERROR] Gagal membaca vibes_caption.csv: {e}")
        return []

vibes_list = load_vibes("vibes_caption.csv")

# === Sentence Transformer Setup ===
similarity_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vibes_embeddings = similarity_model.encode(vibes_list, convert_to_tensor=True)
used_vibes_map = {}

# === Preprocessing Helper ===
def preprocess(text):
    stopwords = {"yang", "di", "ke", "dengan", "dan", "atau", "dari", "untuk", "pada", "adalah", "itu", "ini", "sebuah", "seorang"}
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return [word for word in text.split() if word not in stopwords]

# === Semantic Vibes Matching ===
def match_vibes_semantic(translated_caption, filename):
    try:
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
    except Exception as e:
        print(f"[ERROR] Matching vibes: {e}")
    
    return random.choice(vibes_list)

# === Generate Caption from Image ===
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] Caption generation: {e}")
        return "Tidak dapat menghasilkan caption."

# === Translate Caption ===
def translate_caption(text):
    try:
        return translator.translate(text, src='en', dest='id').text
    except Exception as e:
        print(f"[ERROR] Translasi: {e}")
        return text

# === Routes ===
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        files = request.files.getlist("images")
        for file in files:
            if file and file.filename:
                filename = f"{random.randint(1000, 9999)}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                caption_en = generate_caption(filepath)
                translated = translate_caption(caption_en)
                final_caption = match_vibes_semantic(translated, filename)

                results.append({
                    "image": filepath,
                    "caption_en": caption_en,
                    "translated": translated,
                    "final_caption": final_caption,
                    "filename": filename
                })
    return render_template("index.html", results=results)

@app.route("/regenerate", methods=["POST"])
def regenerate():
    try:
        filename = request.form.get("filename")
        translated = request.form.get("translated")

        if not filename or not translated:
            return jsonify({"error": "Data tidak lengkap"}), 400

        new_caption = match_vibes_semantic(translated, filename)
        return jsonify({"final_caption": new_caption})
    except Exception as e:
        print(f"[ERROR] Regenerate: {e}")
        return jsonify({"error": "Terjadi kesalahan saat generate ulang"}), 500

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
