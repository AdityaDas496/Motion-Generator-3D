import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MotionDataset, MotionDataCollator # From dataset.py
from model import MotionTransformer                  # From model.py

# --- 1. --- SETTINGS (YOU CAN CHANGE THESE) ---
BASE_PATH = r"C:/Users/Aditya Das/Downloads/Blender Project Final"
BATCH_SIZE = 1       # How many animations to look at at once
NUM_EPOCHS = 100     # How many times to loop through the *entire* dataset
LEARNING_RATE = 1e-6 # How "fast" the model should learn
SAVE_EVERY_N_EPOCHS = 10 # How often to save a backup of your model

# --- 2. --- SETUP FILE PATHS ---
MOTION_FILE = os.path.join(BASE_PATH, "kit_motion_data.npy")
TEXT_FILE = os.path.join(BASE_PATH, "kit_text_data.json")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "kit_model.pth")

print("--- STARTING MODEL TRAINING ---")

# --- 3. --- SETUP DATALOADER ---
print("Setting up data loaders...")
dataset = MotionDataset(motion_file=MOTION_FILE, text_file=TEXT_FILE)
collator = MotionDataCollator()
data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       
    collate_fn=collator 
)

# --- 4. --- SETUP MODEL ---
# Get the feature count from the first item
first_item = dataset[0]
num_motion_features = first_item['motion'].shape[-1]
print(f"Motion features detected: {num_motion_features}")

model = MotionTransformer(motion_features=num_motion_features)

# --- 5. --- SETUP LOSS AND OPTIMIZER ---
# Loss Function: How we measure "error"
# We'll use Mean Squared Error: the average squared difference
# between the model's guess and the real animation.
loss_function = nn.MSELoss()

# Optimizer: How the model "learns"
# AdamW is a modern, powerful optimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("Setup complete. Starting training loop...")

# --- 6. --- THE TRAINING LOOP ---
for epoch in range(NUM_EPOCHS):
    
    # We'll track the average loss for this epoch
    total_loss = 0.0
    
    # Go through each batch in our dataloader
    for i, batch in enumerate(data_loader):
        
        motion_batch = batch['motion_batch']
        text_batch = batch['text_batch']
        
        # --- The Core 4 Steps of Training ---
        
        # 1. Clear old errors (gradients)
        optimizer.zero_grad()
        
        # 2. Forward Pass: Get the model's prediction
        predicted_motion = model(text_batch, motion_batch)
        
        # 3. Calculate Loss: How wrong was the prediction?
        # We compare the model's guess (predicted_motion) 
        # to the actual animation (motion_batch).
        loss = loss_function(predicted_motion, motion_batch)
        
        # 4. Backward Pass: Calculate how to fix the error
        loss.backward()

        # --- NEW LINE ---
        # "Clip" the gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # --- END NEW LINE ---
        
        # 5. Step: Update the model's "brain"
        optimizer.step()
        
        total_loss += loss.item() # .item() gets the raw number
        
    # --- End of Epoch ---
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Average Loss: {avg_loss:.6f}")
    
    # Save a checkpoint
    if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"--- Model saved to {MODEL_SAVE_PATH} ---")

# --- 7. --- FINAL SAVE ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("--- TRAINING COMPLETE ---")
print(f"Final model saved to {MODEL_SAVE_PATH}")