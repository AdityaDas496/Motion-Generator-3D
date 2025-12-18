import os
import torch
from torch.utils.data import DataLoader
from dataset import MotionDataset, MotionDataCollator # Import from dataset.py
from model import MotionTransformer                  # Import from model.py

# --- 1. --- YOU MUST EDIT THIS ---
BASE_PATH = r"C:/Users/Aditya Das/Downloads/Blender Project Final"

MOTION_FILE = os.path.join(BASE_PATH, "kit_motion_data.npy")
TEXT_FILE = os.path.join(BASE_PATH, "kit_text_data.json")

print("--- TESTING THE MODEL ARCHITECTURE ---")

# --- 2. --- Get one batch of REAL data ---
# We need a real batch to know the exact "features" dimension
try:
    dataset = MotionDataset(motion_file=MOTION_FILE, text_file=TEXT_FILE)
    collator = MotionDataCollator()
    data_loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    
    print("Getting one batch...")
    batch = next(iter(data_loader))
    
    motion_batch = batch['motion_batch']
    text_batch = batch['text_batch']
    
    # This is CRITICAL! We get the feature count from the data itself.
    # It will be 221 (or whatever your data has)
    num_motion_features = motion_batch.shape[-1] 
    
    print(f"Data batch loaded. Motion features detected: {num_motion_features}")

except Exception as e:
    print(f"❌ ERROR: Failed to load data batch. {e}")
    exit()

# --- 3. --- Initialize the Model ---
print("\nInitializing the MotionTransformer model...")
try:
    # We pass our dynamically-found feature count to the model
    model = MotionTransformer(motion_features=num_motion_features)
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval() 
    
    print("✅ Model initialized successfully.")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize model. {e}")
    exit()

# --- 4. --- Run the "Sanity Check" (Forward Pass) ---
print("\nAttempting one forward pass (pushing data through the model)...")
try:
    with torch.no_grad(): # We're not training, so no gradients needed
        output = model(text_batch, motion_batch)
    
    print("\n✅ --- SUCCESS! --- ✅")
    print("Model forward pass complete.")
    print(f"\nInput motion shape: {list(motion_batch.shape)}")
    print(f"Output motion shape: {list(output.shape)}")
    
    if motion_batch.shape == output.shape:
        print("\nShape test PASSED. Input and output shapes match.")
    else:
        print(f"\n❌ SHAPE TEST FAILED. Input: {motion_batch.shape}, Output: {output.shape}")

except Exception as e:
    print(f"❌ ERROR: Model forward pass failed. {e}")