import os
from torch.utils.data import DataLoader
from dataset import MotionDataset, MotionDataCollator # Import from our new file

# Make sure this is the correct path to your project folder
BASE_PATH = r"PUT_PATH_TO_PROJECT_FOLDER"

MOTION_FILE = os.path.join(BASE_PATH, "kit_motion_data.npy")
TEXT_FILE = os.path.join(BASE_PATH, "kit_text_data.json")

print("TESTING THE DATASET AND DATALOADER")

# Initialize the dataset
# This will load the .npy and .json files
try:
    dataset = MotionDataset(motion_file=MOTION_FILE, text_file=TEXT_FILE)
except Exception as e:
    print(f"Error initializing dataset: {e}")
    exit()

# Initialize Collator
# Downloading CLIP Model
try:
    collator = MotionDataCollator()
except Exception as e:
    print(f"Error initializing collator (CLIP): {e}")
    print("   This can fail if you are offline.")
    exit()

# Initialize Dataloader
# This bundles the dataset and collator
# We'll use a batch size of 4
data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,       # Shuffle the data every epoch
    collate_fn=collator # Use our custom "batching" function
)

# Get one batch
print("\nAttempting to get one batch from the DataLoader...")
try:
    # This calls dataset.__getitem__ 4 times, then collator.__call__ once
    first_batch = next(iter(data_loader))
    
    print("\nSUCCESS!")
    print("Successfully retrieved one batch of data.")
    
    # Check the shapes
    motion = first_batch['motion_batch']
    text = first_batch['text_batch']
    
    print(f"\nMotion Batch Shape: {list(motion.shape)}")
    print("   (Batch Size, Max Frames, Features)")
    
    print(f"\nText Batch 'input_ids' Shape: {list(text.input_ids.shape)}")
    print("   (Batch Size, Max Text Tokens)")

except Exception as e:
    print(f"Error getting batch from DataLoader: {e}")
