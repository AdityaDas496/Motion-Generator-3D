import os
import json
import xml.etree.ElementTree as ET # This is a built-in library for reading XML
import numpy as np

print("Script is running...")

# --- 1. --- YOU MUST EDIT THIS ---
# Point this to the folder you unzipped (the one with all the numbered files)
KIT_DATA_PATH = r"C:/Users/Aditya Das/Downloads/2017-06-22"

# --- 2. --- SETTINGS ---
# Let's just load the first 50 motions for a test
MOTIONS_TO_LOAD = 50 

# Where to save the final files
SAVE_PATH = r"C:/Users/Aditya Das/Downloads/Blender Project Final" # Your new project folder
MOTION_SAVE_FILE = os.path.join(SAVE_PATH, "kit_motion_data.npy")
TEXT_SAVE_FILE = os.path.join(SAVE_PATH, "kit_text_data.json")

# --- 3. --- MAIN SCRIPT ---

def get_all_motion_ids(path):
    """Finds all unique motion IDs in the folder."""
    files = os.listdir(path)
    # Get the first 5 digits from all files, and get a unique set
    all_ids = sorted(list(set([f[:5] for f in files])))
    return all_ids

def parse_motion_xml(xml_file):
    """
    Parses the _mmm.xml file and extracts the motion data.
    --- V2: Corrected with the right tag names ---
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        motion_node = root.find(".//Motion[JointOrder]")
        if motion_node is None:
            return None 

        # Get all frame nodes
        frame_nodes = motion_node.findall("MotionFrames/MotionFrame")
        
        # Get the joint order *once*
        joint_order_nodes = motion_node.findall("JointOrder/Joint")
        joint_names = [j.get("name") for j in joint_order_nodes]
        
        all_frames_data = []

        for frame in frame_nodes:
            frame_data_list = []
            
            # --- 1. Get Root Position ---
            root_pos_node = frame.find("RootPosition")
            if root_pos_node is None:
                continue # Skip this frame if it's missing data
            root_pos = [float(p) for p in root_pos_node.text.split()]
            frame_data_list.extend(root_pos)
            
            # --- 2. Get Root Rotation ---
            root_rot_node = frame.find("RootRotation")
            if root_rot_node is None:
                continue # Skip this frame
            root_rot = [float(r) for r in root_rot_node.text.split()]
            frame_data_list.extend(root_rot)
            
            # --- 3. Get all joint rotations ---
            # THIS IS THE FIX: It's <JointPosition>, not <JointAngles>
            joint_pos_node = frame.find("JointPosition") 
            if joint_pos_node is None:
                continue # Skip this frame
            joint_pos = [float(a) for a in joint_pos_node.text.split()]
            frame_data_list.extend(joint_pos)
            
            all_frames_data.append(frame_data_list)
            
        if not all_frames_data:
            return None, None # File was empty
            
        return np.array(all_frames_data), joint_names

    except Exception as e:
        print(f"  ...Error parsing XML {xml_file}: {e}")
        return None, None

def main():
    print("--- STARTING NEW KIT-ML DATA EXTRACTOR ---")
    
    # These will hold our final data
    all_motion_data = []
    all_text_data = []
    
    all_ids = get_all_motion_ids(KIT_DATA_PATH)
    print(f"Found {len(all_ids)} total motions. Loading first {MOTIONS_TO_LOAD}...")
    
    processed_count = 0
    joint_names_list = None # To store the skeleton
    
    for motion_id in all_ids[:MOTIONS_TO_LOAD]:
        
        # Define the 4 files for this ID
        text_file = os.path.join(KIT_DATA_PATH, f"{motion_id}_annotations.json")
        xml_file = os.path.join(KIT_DATA_PATH, f"{motion_id}_mmm.xml")
        
        # 1. Read the text annotations
        try:
            with open(text_file, 'r') as f:
                annotations = json.load(f)
            
            if not annotations:
                # This motion has no text labels, skip it
                print(f"Skipping {motion_id} (no annotations)")
                continue
                
        except Exception as e:
            print(f"Error reading JSON {text_file}: {e}")
            continue

        # 2. Read the motion data
        motion_array, joint_names = parse_motion_xml(xml_file)
        
        if motion_array is None:
            print(f"Skipping {motion_id} (could not parse XML)")
            continue
            
        if joint_names_list is None:
            joint_names_list = joint_names
            print(f"--- DETECTED SKELETON ({len(joint_names)} JOINTS) ---")
            print(joint_names)
            print("--------------------------------------------------")

        # 3. Add to our master list
        # For *each* text annotation, we add the *same* motion
        for text in annotations:
            all_motion_data.append(motion_array)
            all_text_data.append(text)
            
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{MOTIONS_TO_LOAD} motions...")

    print("\n--- PROCESSING COMPLETE ---")
    print(f"Successfully processed {processed_count} motions.")
    print(f"Created {len(all_text_data)} total (motion, text) pairs.")

    # --- 4. SAVE FINAL FILES ---
    if not all_motion_data:
        print("FATAL ERROR: No motion data was extracted.")
        return
        
    # Ensure the save directory exists
    os.makedirs(SAVE_PATH, exist_ok=True)
        
    np.save(MOTION_SAVE_FILE, np.array(all_motion_data, dtype=object))
    print(f"Motion data saved to: {MOTION_SAVE_FILE}")
    
    with open(TEXT_SAVE_FILE, 'w') as f:
        json.dump(all_text_data, f, indent=4)
    print(f"Text data saved to: {TEXT_SAVE_FILE}")
    
    print("\n--- NEW PIPELINE V2 (EXTRACTOR) COMPLETE! ---")


# --- Run the script ---
if __name__ == "__main__":
    main()