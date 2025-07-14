from googletrans import Translator

translator = Translator()

def translate_caption(text):
    try:
        return translator.translate(text, src='en', dest='id').text
    except Exception as e:
        print(f"[ERROR] Translasi: {e}")
        return text
