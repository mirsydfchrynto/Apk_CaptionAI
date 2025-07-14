from flask import Flask, render_template, request, jsonify
import os, random
from config import UPLOAD_FOLDER
from generate_caption import generate_caption
from translate_caption import translate_caption
from match_vibes import match_vibes_semantic, used_vibes_map
from load_vibes import vibes_list

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

if __name__ == "__main__":
    app.run(debug=True)
