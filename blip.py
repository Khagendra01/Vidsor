from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import time

# --- choose the lightweight Tiny-BLIP checkpoint
model_id = "Salesforce/blip-image-captioning-base"

# --- load model + processor (uses ~4 GB VRAM)
start_time = time.time()
processor = BlipProcessor.from_pretrained(model_id, use_fast=True)

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = BlipForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",          # will offload to CPU if GPU full
    quantization_config=quantization_config,  # optional, needs bitsandbytes
    use_safetensors=True       # use safetensors to avoid torch.load vulnerability
)
load_time = time.time() - start_time
print(f"⏱️  Model loading time: {load_time:.2f} seconds")

# --- load an image
image = Image.open("hiker.png").convert("RGB")

# --- preprocess
inputs = processor(image, return_tensors="pt").to(model.device)

# --- generate caption
start_time = time.time()
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=50)
generation_time = time.time() - start_time

caption = processor.decode(output[0], skip_special_tokens=True)
print("🧠 Generated Caption:", caption)
print(f"⏱️  Caption generation time: {generation_time:.2f} seconds")
