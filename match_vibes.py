from sentence_transformers import SentenceTransformer, util
from load_vibes import vibes_list
import random

similarity_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vibes_embeddings = similarity_model.encode(vibes_list, convert_to_tensor=True)
used_vibes_map = {}

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
