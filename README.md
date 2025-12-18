# Generative 3D motion generator

This is a project where I'm trying to generate **human motion** (like walking, turning, etc.) directly from **text descriptions** using a Transformer model.
Basically: you write something like *"a person walks forward"* and the model outputs a motion sequence.

I’m still learning Python, PyTorch, and generative models while building this, so the codebase is a mix of experiments, debugging scripts, and the actual working pipeline.

This repo contains:

* A script for extracting motion data + text from the **KIT Motion-Language** dataset
* A PyTorch **Dataset + DataLoader** that uses CLIP to encode text
* A custom **Transformer model** that predicts motion frame-by-frame
* Training and generation scripts
* Some test/debug tools I made while figuring things out

If you're also new to AI or motion-generation, this might help you get started.

---

## Project Overview (Simple Version)

Here’s the whole idea in one diagram-ish list:

```
KIT dataset (.xml + text) 
         ↓
data extraction → motion.npy + text.json
         ↓
dataset + collator (pads motion, encodes text)
         ↓
Transformer (CLIP text + motion features)
         ↓
training (MSE loss)
         ↓
generation from a text prompt
```

At the end, you get something like:

```
kit_generated_motion.npy
```

which contains a motion sequence you can visualize later (Blender, Python plotting, etc.).

---

## Files in This Repo

| File                  | What it does                                                       |
| --------------------- | ------------------------------------------------------------------ |
| `load_amass_data.py`  | Extracts KIT XML files → NumPy arrays of motion + text annotations |
| `dataset.py`          | Handles loading, padding, and CLIP text encoding                   |
| `model.py`            | The actual Transformer model                                       |
| `train.py`            | Training loop (MSE loss, AdamW, gradient clipping)                 |
| `generate_v2.py`      | Generate new motion from a text prompt                             |
| `test_dataset.py`     | Sanity check: does the dataloader work?                            |
| `test_model.py`       | Does the model forward-pass without crashing?                      |
| `debug_xml_parser.py` | Prints problematic KIT XML files for debugging                     |

All actual dataset files (`kit_motion_data.npy`, `kit_text_data.json`) get created after you run the extractor.

---

## Setup

1. Clone the project

   ```bash
   git clone <your repo>
   cd <repo>
   ```

2. Create virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install required packages

   ```bash
   pip install torch transformers numpy tqdm
   ```

4. Download the KIT ML dataset and set the folder path in:

   ```
   load_amass_data.py → KIT_DATA_PATH
   ```

---

## Step 1: Extract the dataset

Run:

```bash
python load_amass_data.py
```

It will:

* parse KIT .xml motion files
* get joint positions per frame
* pair motions with text annotations
* save everything as:

```
kit_motion_data.npy
kit_text_data.json
```

If something fails, `debug_xml_parser.py` prints out the problematic XML.

---

## Step 2: Test that things work

Run:

```bash
python test_dataset.py
python test_model.py
```

If both run without errors, training should work fine.

---

## Step 3: Train the model

```bash
python train.py
```

This will:

* load the dataset
* run MSE training on variable-length sequences
* save the model as `kit_model.pth`

You can tweak:

```python
NUM_EPOCHS
LEARNING_RATE
BATCH_SIZE
```

inside `train.py`.

---

## Step 4: Generate motion from text

Edit the prompt inside `generate_v2.py`:

```python
PROMPT = "A person walks forward."
```

Then run:

```bash
python generate_v2.py
```

Output:

```
kit_generated_motion.npy
```

which contains predicted motion frames.

---

## Notes / Things I Still Need to Improve

* Add **masked loss** (so padded frames don’t affect training)
* Normalize/denormalize motion features
* Add visualization tools (Blender import or Python skeletal animation)
* Possibly switch to cross-attention instead of simply adding text embeddings
* Try more advanced generation (autoregressive or diffusion-style)

Since I’m still learning, the code is evolving, but the current version works end-to-end for generating basic text-conditioned motion.

---

## If you’re trying to learn from this

Feel free to use this repo as a reference if you're learning:

* PyTorch
* Transformers
* CLIP text embeddings
* Sequence models
* Motion datasets
