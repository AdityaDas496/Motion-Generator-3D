import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MotionDataset, MotionDataCollator # From dataset.py
from model import MotionTransformer                  # From model.py

# Setting values of batch size, number of epochs learning rate, etc
BASE_PATH = r"PATH_TO_PROJECT_FOLDER"
BATCH_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
SAVE_EVERY_N_EPOCHS = 10

# Set file path
MOTION_FILE = os.path.join(BASE_PATH, "kit_motion_data.npy")
TEXT_FILE = os.path.join(BASE_PATH, "kit_text_data.json")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "kit_model.pth")

print("STARTING MODEL TRAINING")

# Set up data loaders
print("Setting up data loaders...")
dataset = MotionDataset(motion_file=MOTION_FILE, text_file=TEXT_FILE)
collator = MotionDataCollator()
data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       
    collate_fn=collator 
)

# Model setup
# Get the feature count from the first item
first_item = dataset[0]
num_motion_features = first_item['motion'].shape[-1]
print(f"Motion features detected: {num_motion_features}")

model = MotionTransformer(motion_features=num_motion_features)

# Set loss and optimizer
# Loss Function: How we measure "error"
# We'll use Mean Squared Error: the average squared difference
# between the model's guess and the real animation.
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("Setup complete. Starting training loop...")

# Training loop
for epoch in range(NUM_EPOCHS):
    
    # We'll track the average loss for this epoch
    total_loss = 0.0
    
    # Go through each batch in our dataloader
    for i, batch in enumerate(data_loader):
        
        motion_batch = batch['motion_batch']
        text_batch = batch['text_batch']
        optimizer.zero_grad()
        predicted_motion = model(text_batch, motion_batch)
        loss = loss_function(predicted_motion, motion_batch)
        loss.backward()
        # Clip the gradients to prevent them from exploding (Pun intended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)-
        optimizer.step()
        
        total_loss += loss.item() # .item() gets the raw number
        
    # End of Epoch
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Average Loss: {avg_loss:.6f}")
    
    # Save a checkpoint
    if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

# Finally save it
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("TRAINING COMPLETE")
print(f"Final model saved to {MODEL_SAVE_PATH}")
