import os

# Point this to the folder you unzipped
# Keep mind of the path of the where you store the AMASS dataset
KIT_DATA_PATH = r"PUT_PATH_HERE"

# Define the file to check
# We'll just look at the first file that failed
XML_FILE_PATH = os.path.join(KIT_DATA_PATH, "00001_mmm.xml")

print(f"DEBUGGING XML FILE: {XML_FILE_PATH}")

try:
    with open(XML_FILE_PATH, 'r') as f:
        # Read and print the first 20 lines
        for i in range(20):
            line = f.readline()
            if not line:
                break # Stopping in case the file is shorter than 20 lines
            print(line, end='') # 'end='' stops it from double-spacing

    print("\n\nDEBUGGING COMPLETE")

except FileNotFoundError:
    print(f"Error: File not found. Check your KIT_DATA_PATH.")
except Exception as e:
    print(f"ERROR: {e}")
