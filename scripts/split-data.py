# scripts/split_dataset.py
import json
from sklearn.model_selection import train_test_split
import os

# Define paths
input_path = 'data/training/epfo_qa_dataset.json'
train_output_path = 'data/training/train_dataset.json'
val_output_path = 'data/training/val_dataset.json'

# Ensure the input file exists
try:
    with open(input_path, 'r') as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {input_path}. Please run the generation script first.")
    exit()

# Perform the split
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Save the splits
os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
with open(train_output_path, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(val_output_path, 'w') as f:
    json.dump(val_data, f, indent=4)

print(f"Data split into {len(train_data)} training samples and {len(val_data)} validation samples.")