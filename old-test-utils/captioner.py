import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import torchvision.transforms as transforms

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
# Use local model if available, otherwise fall back to Hugging Face
script_dir = os.path.dirname(os.path.abspath(__file__))
local_model_path = os.path.join(script_dir, "models", "ShareCaptioner-Video")
if os.path.exists(local_model_path):
    model_name = local_model_path
    print(f"Using local model: {model_name}")
else:
    model_name = "Lin-Chen/ShareCaptioner-Video"
    print(f"Local model not found, using Hugging Face: {model_name}")

# how many frames to sample from the video
NUM_FRAMES = 6          # fewer frames -> lower VRAM use
FRAME_SIZE = (224, 224) # 224px resolution for 3060Ti
video_path = "kbc.mp4"

# -------------------------------------------------------------
# 1. Load model + tokenizer (based on official app.py)
# -------------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Use AutoModel with 4-bit quantization for 8GB GPU
if torch.cuda.is_available():
    try:
        # Load without quantization (quantization incompatible with im_mask parameter)
        # Use float16 to reduce memory usage
        print("Loading model in float16 (quantization incompatible with im_mask)...")
        import tempfile
        offload_folder = tempfile.mkdtemp()
        model = AutoModel.from_pretrained(
            model_name, 
            dtype=torch.float16,
            device_map="auto",    # Automatically handles GPU/CPU/disk offloading
            offload_folder=offload_folder,  # For disk offloading if needed
            trust_remote_code=True
        ).eval()
        model.tokenizer = tokenizer
        print("Model loaded in float16 (may use more memory)")
    except Exception as e:
        print(f"Error loading with 4-bit quantization: {e}")
        print("Trying 8-bit quantization...")
        try:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModel.from_pretrained(
                model_name, 
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            model.tokenizer = tokenizer
            print("Model loaded with 8-bit quantization")
        except Exception as e2:
            print(f"Error loading with 8-bit quantization: {e2}")
            print("Trying without quantization (may OOM)...")
            model = AutoModel.from_pretrained(
                model_name, 
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            model.tokenizer = tokenizer
            print("Model loaded without quantization (may cause OOM)")
else:
    model = AutoModel.from_pretrained(
        model_name, 
        dtype=torch.float32, 
        device_map="cpu",
        trust_remote_code=True
    ).eval()
    model.tokenizer = tokenizer
    print("Model loaded on CPU")

# Patch prepare_inputs_for_generation to handle None past_key_values
# This fixes the AttributeError when using inputs_embeds
original_prepare = model.prepare_inputs_for_generation
def patched_prepare_inputs_for_generation(self, input_ids=None, past_key_values=None, **kwargs):
    # When using inputs_embeds, always set past_key_values to None to avoid structure issues
    if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
        past_key_values = None
    # Also check if past_key_values has invalid structure (None elements that would cause errors)
    elif past_key_values is not None:
        try:
            # Validate structure - if past_key_values[0][0] is None, it will cause the error
            if len(past_key_values) > 0 and past_key_values[0] is not None:
                if len(past_key_values[0]) > 0 and past_key_values[0][0] is None:
                    past_key_values = None
        except (AttributeError, TypeError, IndexError):
            past_key_values = None
    return original_prepare(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
model.prepare_inputs_for_generation = patched_prepare_inputs_for_generation.__get__(model, type(model))

# Patch _validate_model_kwargs to accept im_mask
original_validate = model._validate_model_kwargs
def patched_validate_model_kwargs(self, model_kwargs):
    # Remove im_mask from validation - it's a valid parameter for this model
    if 'im_mask' in model_kwargs:
        im_mask = model_kwargs.pop('im_mask')
        result = original_validate(model_kwargs)
        model_kwargs['im_mask'] = im_mask  # Put it back
        return result
    return original_validate(model_kwargs)
model._validate_model_kwargs = patched_validate_model_kwargs.__get__(model, type(model))

# Patch _prepare_model_inputs to accept inputs_embeds
# The model DOES support inputs_embeds, but transformers validation rejects it
original_prepare_model_inputs = model._prepare_model_inputs
def patched_prepare_model_inputs(self, inputs, bos_token_id=None, model_kwargs=None):
    # model_kwargs is a dict that may contain inputs_embeds
    if model_kwargs is None:
        model_kwargs = {}
    
    # If inputs_embeds is provided in model_kwargs, bypass validation completely
    if 'inputs_embeds' in model_kwargs and model_kwargs['inputs_embeds'] is not None:
        # Extract inputs_embeds and return it directly, removing it from model_kwargs
        inputs_embeds = model_kwargs.pop('inputs_embeds')
        # Create dummy input_ids tensor for generate() method compatibility
        # generate() needs input_ids even when using inputs_embeds for the first step
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        # Use pad_token_id or 0 as dummy token IDs (won't be used since we have inputs_embeds)
        pad_token_id = getattr(self.config, 'pad_token_id', 0) or 0
        dummy_input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long, device=inputs_embeds.device)
        model_kwargs['input_ids'] = dummy_input_ids
        return inputs_embeds, 'inputs_embeds', model_kwargs
    
    # Otherwise use the original method
    return original_prepare_model_inputs(inputs, bos_token_id=bos_token_id, model_kwargs=model_kwargs)
model._prepare_model_inputs = patched_prepare_model_inputs.__get__(model, type(model))

# Also need to ensure the model's forward signature check passes
# Patch the method that checks if inputs_embeds is supported
if hasattr(model, '_get_model_inputs'):
    original_get_model_inputs = model._get_model_inputs
    def patched_get_model_inputs(self, *args, **kwargs):
        # Force return that inputs_embeds is supported
        result = original_get_model_inputs(*args, **kwargs)
        return result
    model._get_model_inputs = patched_get_model_inputs.__get__(model, type(model))

# -------------------------------------------------------------
# 2. Helper functions (from official app.py)
# -------------------------------------------------------------
def padding_336(b, pad=336):
    width, height = b.size
    tar = int(np.ceil(height / pad) * pad)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255])
    return b

def HD_transform(img, hd_num=25):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * 336)
    new_h = int(new_w / ratio)
    img = transforms.functional.resize(img, [new_h, new_w],)
    img = padding_336(img, 336)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)
    return img

# -------------------------------------------------------------
# 3. Extract keyframes
# -------------------------------------------------------------
def sample_frames(video_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

frames = sample_frames(video_path, NUM_FRAMES)
print(f"Extracted {len(frames)} frames")

# -------------------------------------------------------------
# 4. Concatenate frames into a single image grid
# -------------------------------------------------------------
concat_img = Image.new("RGB", (FRAME_SIZE[0], FRAME_SIZE[1] * len(frames)))
for i, img in enumerate(frames):
    concat_img.paste(img, (0, i * FRAME_SIZE[1]))

# -------------------------------------------------------------
# 5. Inference using model.chat() method (simpler, like InternLM example)
# -------------------------------------------------------------
prompt = "Here are a few key frames of a video, <ImageHere> describe this video in detail."

# Save concatenated image to temporary file
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
    concat_img.save(tmp_file.name)
    image_path = tmp_file.name

print("Generating caption...")
# Use model.chat() method directly (like InternLM-XComposer example)
# This method handles all the internal complexity for us
if torch.cuda.is_available():
    with torch.amp.autocast('cuda'):
        caption, history = model.chat(
            tokenizer,
            query=prompt,
            image=image_path,  # Can be file path or PIL Image
            max_new_tokens=512,
            do_sample=False,
            num_beams=3
        )
else:
    with torch.no_grad():
        caption, history = model.chat(
            tokenizer,
            query=prompt,
            image=image_path,
            max_new_tokens=512,
            do_sample=False,
            num_beams=3
        )

# Clean up temporary file
os.unlink(image_path)

print("\n--- Generated Caption ---")
print(caption)
