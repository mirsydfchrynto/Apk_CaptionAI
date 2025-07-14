from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] Caption generation: {e}")
        return "Tidak dapat menghasilkan caption."
