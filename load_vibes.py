import csv

def load_vibes(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [row['vibes'] for row in reader if 'vibes' in row and row['vibes'].strip()]
    except Exception as e:
        print(f"[ERROR] Gagal membaca vibes_caption.csv: {e}")
        return []

vibes_list = load_vibes("vibes_caption.csv")
