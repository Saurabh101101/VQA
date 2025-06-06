import pickle

# Path to your processed file
processed_data_path = "D:/WORK/PG/Project/vqa_project/data/processed/image_features_val.pkl"

# Load the processed dataset
with open(processed_data_path, "rb") as f:
    processed_data = pickle.load(f)

# Check the size of the processed data
print(f"Total number of entries in processed dataset: {len(processed_data)}")

# Preview a few samples
for i, (key, value) in enumerate(processed_data.items()):
    print(f"Sample {i+1}:")
    print(f"  Image ID: {key}")
    print(f"  Data: {value}")
    if i == 4:  # Limit output to first 5 samples
        break
# for item in processed_data[:3]:  # Access first 3 items
#     for key, value in item.items():  # Access key-value pairs in each dictionary
#         print(key, value)