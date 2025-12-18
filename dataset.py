import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.nn.utils.rnn import pad_sequence

# Define main dataset class
class MotionDataset(Dataset):
    def __init__(self, motion_file, text_file):
        print(f"Loading data from {motion_file} and {text_file}...")
        
        # Load the motion data
        self.motion_data = np.load(motion_file, allow_pickle=True)
        
        # Load the text data
        with open(text_file, 'r') as f:
            self.text_data = json.load(f)
            
        # Make sure they match
        assert len(self.motion_data) == len(self.text_data)
        
        print(f"Data loaded. Total samples: {len(self.text_data)}")

    def __len__(self):
        # How many items are in the dataset?
        return len(self.text_data)

    def __getitem__(self, idx):
        # Get one item at a time
        
        # Get the motion and convert to a PyTorch tensor
        motion = self.motion_data[idx]
        motion_tensor = torch.tensor(motion, dtype=torch.float32)
        
        # Get the raw text description
        text = self.text_data[idx]
        
        # The raw text is returned here. The "collate" function will process it.
        return {
            "motion": motion_tensor,
            "text": text
        }

# Define Batching function
class MotionDataCollator:
    def __init__(self):
        # Load the CLIP model and processor once
        print("Initializing CLIP processor...")
        # This will download the model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def __call__(self, batch):
        # This function is called by the DataLoader
        
        # Separate the motions and text
        motions_list = [item['motion'] for item in batch]
        text_list = [item['text'] for item in batch]
        
        # Process Text
        # The processor turns all text strings into numerical tokens
        text_inputs = self.processor(
            text=text_list, 
            return_tensors="pt",  # Return PyTorch tensors
            padding=True,         # Pad all text to the same length
            truncation=True       # Truncate text if it's too long
        )
        
        # Process Motion
        # We pad the motion sequences to be the same length
        # 'batch_first=True' makes the shape [batch_size, seq_len, features]
        motions_padded = pad_sequence(motions_list, batch_first=True, padding_value=0.0)
        
        return {
            "motion_batch": motions_padded,
            "text_batch": text_inputs
        }
