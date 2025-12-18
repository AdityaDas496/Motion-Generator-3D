import os
import torch
from torch.utils.data import DataLoader
from dataset import MotionDataset, MotionDataCollator # Import from dataset.py
from model import MotionTransformer                  # Import from model.py

BASE_PATH = r"PATH_TO_PROJECT_FOLDER"

MOTION_FILE = os.path.join(BASE_PATH, "kit_motion_data.npy")
TEXT_FILE = os.path.join(BASE_PATH, "kit_text_data.json")

print("TESTING THE MODEL ARCHITECTURE")

# Get one batch of REAL data
# We need a real batch to know the exact "features" dimension
try:
    dataset = MotionDataset(motion_file=MOTION_FILE, text_file=TEXT_FILE)
    collator = MotionDataCollator()
    data_loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    
    print("Getting one batch...")
    batch = next(iter(data_loader))
    
    motion_batch = batch['motion_batch']
    text_batch = batch['text_batch']
    
    # Get the feature count from the dataset
    num_motion_features = motion_batch.shape[-1] 
    
    print(f"Data batch loaded. Motion features detected: {num_motion_features}")

except Exception as e:
    print(f"Error: Failed to load data batch. {e}")
    exit()

# Initialize the model
print("\nInitializing the MotionTransformer model...")
try:
    # We pass the dynamically-found feature count to the model
    model = MotionTransformer(motion_features=num_motion_features)
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval() 
    
    print("Model initialized successfully.")
except Exception as e:
    print(f"Error: Failed to initialize model. {e}")
    exit()

# Sanity Check
print("\nAttempting one forward pass (pushing data through the model)...")
try:
    with torch.no_grad():
        output = model(text_batch, motion_batch)
    
    print("\nSUCCESS!")
    print("Model forward pass complete.")
    print(f"\nInput motion shape: {list(motion_batch.shape)}")
    print(f"Output motion shape: {list(output.shape)}")
    
    if motion_batch.shape == output.shape:
        print("\nShape test passed. Input and output shapes match.")
    else:
        print(f"\nShape test failed. Input: {motion_batch.shape}, Output: {output.shape}")

except Exception as e:
    print(f"Error: Model forward pass failed. {e}")
