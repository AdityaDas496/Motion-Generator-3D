import os
from torch.utils.data import DataLoader
from dataset import MotionDataset, MotionDataCollator # Import from our new file

# --- 1. --- YOU MUST EDIT THIS ---
# Make sure this is the correct path to your project folder
BASE_PATH = r"C:/Users/Aditya Das/Downloads/Blender Project Final"

MOTION_FILE = os.path.join(BASE_PATH, "kit_motion_data.npy")
TEXT_FILE = os.path.join(BASE_PATH, "kit_text_data.json")

print("--- TESTING THE DATASET AND DATALOADER ---")

# --- 2. --- Initialize the Dataset ---
# This will load the .npy and .json files
try:
    dataset = MotionDataset(motion_file=MOTION_FILE, text_file=TEXT_FILE)
except Exception as e:
    print(f"❌ ERROR initializing dataset: {e}")
    exit()

# --- 3. --- Initialize the Collator ---
# This will download the CLIP model (this might take a minute)
try:
    collator = MotionDataCollator()
except Exception as e:
    print(f"❌ ERROR initializing collator (CLIP): {e}")
    print("   This can fail if you are offline.")
    exit()

# --- 4. --- Initialize the DataLoader ---
# This bundles the dataset and collator
# We'll use a batch size of 4
data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,       # Shuffle the data every epoch
    collate_fn=collator # Use our custom "batching" function
)

# --- 5. --- Get one batch ---
print("\nAttempting to get one batch from the DataLoader...")
try:
    # This calls dataset.__getitem__ 4 times, then collator.__call__ once
    first_batch = next(iter(data_loader))
    
    print("\n✅ --- SUCCESS! --- ✅")
    print("Successfully retrieved one batch of data.")
    
    # Check the shapes
    motion = first_batch['motion_batch']
    text = first_batch['text_batch']
    
    print(f"\nMotion Batch Shape: {list(motion.shape)}")
    print("   (Batch Size, Max Frames, Features)")
    
    print(f"\nText Batch 'input_ids' Shape: {list(text.input_ids.shape)}")
    print("   (Batch Size, Max Text Tokens)")

except Exception as e:
    print(f"❌ ERROR getting batch from DataLoader: {e}")