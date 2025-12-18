import os

# --- 1. --- YOU MUST EDIT THIS ---
# Point this to the folder you unzipped
KIT_DATA_PATH = r"C:/Users/Aditya Das/Downloads/2017-06-22"

# --- 2. --- Define the file to check ---
# We'll just look at the first file that failed
XML_FILE_PATH = os.path.join(KIT_DATA_PATH, "00001_mmm.xml")

print(f"--- DEBUGGING XML FILE: {XML_FILE_PATH} ---")

try:
    with open(XML_FILE_PATH, 'r') as f:
        # Read and print the first 20 lines
        for i in range(20):
            line = f.readline()
            if not line:
                break # Stop if the file is shorter than 20 lines
            print(line, end='') # 'end='' stops it from double-spacing

    print("\n\n--- DEBUGGING COMPLETE ---")

except FileNotFoundError:
    print(f"❌ ERROR: File not found. Check your KIT_DATA_PATH.")
except Exception as e:
    print(f"❌ ERROR: {e}")