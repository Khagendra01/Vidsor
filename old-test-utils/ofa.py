from transformers import OFATokenizer, OFAModel
from PIL import Image
from torchvision import transforms
import torch
import time
import os

# Workaround for ETag issue with some Hugging Face models
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Try to import BitsAndBytesConfig (may not be available in all transformers versions)
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    try:
        from bitsandbytes import BitsAndBytesConfig
    except ImportError:
        BitsAndBytesConfig = None

# --- choose the OFA-Large checkpoint
# (official model card: https://huggingface.co/OFA-Sys/OFA-large-caption)
model_id = "OFA-Sys/OFA-large-caption"

# --- load tokenizer + model (uses ~4 GB VRAM)
start_time = time.time()

# Download files using huggingface_hub first to avoid ETag issues
try:
    from huggingface_hub import snapshot_download
    print("📥 Downloading model files using huggingface_hub...")
    snapshot_download(repo_id=model_id, local_files_only=False)
    print("✅ Files downloaded successfully")
    # After downloading, load from local cache to avoid ETag issues
    files_downloaded = True
except ImportError:
    print("⚠️  huggingface_hub not available, will try transformers directly")
    files_downloaded = False
except Exception as e:
    print(f"⚠️  Could not pre-download files: {e}, will try transformers directly")
    files_downloaded = False

# Load tokenizer - use local_files_only if we downloaded with huggingface_hub
if files_downloaded:
    print("📂 Loading tokenizer from local cache...")
    tokenizer = OFATokenizer.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True
    )
else:
    # Fallback: try with transformers (may hit ETag issue)
    try:
        tokenizer = OFATokenizer.from_pretrained(
            model_id,
            local_files_only=False,
            trust_remote_code=True,
            resume_download=False
        )
    except OSError as e:
        if "ETag" in str(e):
            print("⚠️  ETag issue detected. Please install huggingface_hub: pip install huggingface_hub")
            raise
        raise

# Configure 8-bit quantization (if available)
quantization_config = BitsAndBytesConfig(load_in_8bit=True) if BitsAndBytesConfig else None

# Load model with optimizations (quantization may not be fully supported in OFA transformers fork)
model_kwargs = {
    "dtype": torch.float16,
    "device_map": "auto",          # will offload to CPU if GPU full
    "use_safetensors": True,       # use safetensors to avoid torch.load vulnerability
    "use_cache": False
}
if quantization_config is not None:
    model_kwargs["quantization_config"] = quantization_config  # optional, needs bitsandbytes

# Load model - use local_files_only if we downloaded with huggingface_hub
if files_downloaded:
    model_kwargs["local_files_only"] = True
else:
    model_kwargs["local_files_only"] = False

model = OFAModel.from_pretrained(model_id, **model_kwargs)
load_time = time.time() - start_time
print(f"⏱️  Model loading time: {load_time:.2f} seconds")

# --- setup image preprocessing (OFA requires specific transforms)
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# --- load image
image = Image.open("hiker.png").convert("RGB")
patch_img = patch_resize_transform(image).unsqueeze(0).to(model.device)

# --- prepare text input
txt = " what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids.to(model.device)

# --- generate caption
start_time = time.time()
with torch.inference_mode():
    generated_ids = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3, max_new_tokens=50)
generation_time = time.time() - start_time

caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("🧠 Generated Caption:", caption)
print(f"⏱️  Caption generation time: {generation_time:.2f} seconds")
