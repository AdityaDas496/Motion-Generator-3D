import os
import torch
import numpy as np
from transformers import CLIPProcessor
from model import MotionTransformer 
import json

# --- 1. SETTINGS ---
# Use "." to refer to the current folder since the script is inside it
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_PATH, "kit_model.pth")
TEXT_FILE = os.path.join(BASE_PATH, "kit_text_data.json")
MOTION_FILE = os.path.join(BASE_PATH, "kit_motion_data.npy") 

# --- 2. PARAMETERS ---
PROMPT = "A person walks." # <--- You can change this!
NUM_FRAMES = 150                 

print(f"--- GENERATING V2 ANIMATION: '{PROMPT}' ---")

# --- 3. LOAD RESOURCES ---
# Get feature count from real data
try:
    sample = np.load(MOTION_FILE, allow_pickle=True)[0]
    NUM_FEATURES = sample.shape[-1]
    print(f"Detected Features: {NUM_FEATURES}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Load Model
model = MotionTransformer(motion_features=NUM_FEATURES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Load Text Processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- 4. GENERATE ---
try:
    with torch.no_grad():
        # Prepare inputs
        text_inputs = processor(text=[PROMPT], return_tensors="pt", padding=True, truncation=True)
        noise = torch.randn(1, NUM_FRAMES, NUM_FEATURES)
        
        print("Model is thinking...")
        output = model(text_inputs, noise)
        
        # Save result
        output_data = output.squeeze(0).cpu().numpy()
        save_path = os.path.join(BASE_PATH, "kit_generated_motion.npy")
        np.save(save_path, output_data)
        
        print(f"\n✅ SUCCESS! Animation saved to: {save_path}")

except Exception as e:
    print(f"❌ Error: {e}")