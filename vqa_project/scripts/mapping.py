import pickle
import json

# Load the .pkl file
with open("vg_train.pkl", "rb") as f:
    data = pickle.load(f)

# Extract all answers
answers = set(item["answers"] for item in data)  # Using a set to ensure uniqueness

# Create a mapping from unique answers to sequential indices
answer_mapping = {answer: idx for idx, answer in enumerate(sorted(answers))}

# Save as a JSON file
with open("vg_answer_mapping.json", "w") as f:
    json.dump(answer_mapping, f, indent=4)

print("Answer mapping saved as vg_answer_mapping.json")
