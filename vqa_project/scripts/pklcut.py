# import pickle
# from collections import Counter

# # Path to the uploaded file
# file_path = 'vg_test_sorted.pkl'

# # Load the .pkl file
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Assuming data is a list or similar structure containing image IDs
# # Modify 'image_id' to match the actual key or field for image IDs in your data
# if isinstance(data, list):
#     # Extract image IDs and count occurrences
#     image_ids = [item['image_id'] for item in data if 'image_id' in item]
#     image_id_counts = Counter(image_ids)
# else:
#     raise ValueError("Unexpected data structure. Expected a list.")

# # Display the counts for each image ID
# for image_id, count in image_id_counts.items():
#     print(f"Image ID: {image_id}, Count: {count}")





# import pickle
# from collections import defaultdict

# # Load the .pkl file
# input_file = "D:/WORK/PG/Project/vqa_project/data/processed/vg_train_processed.pkl"
# output_file = 'vg_train_trimmed.pkl'

# # Load the pkl file
# def load_pkl(file_path):
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)

# # Save the pkl file
# def save_pkl(data, file_path):
#     with open(file_path, 'wb') as f:
#         pickle.dump(data, f)

# # Function to trim image ID occurrences to a maximum of 3
# def trim_image_ids(data):
#     image_id_counts = defaultdict(int)
#     trimmed_data = []

#     for entry in data:
#         image_id = entry.get('image_id')
#         if image_id:
#             if image_id_counts[image_id] < 3:
#                 trimmed_data.append(entry)
#                 image_id_counts[image_id] += 1

#     return trimmed_data

# # Main processing
# try:
#     # Load data
#     data = load_pkl(input_file)

#     # Trim image ID occurrences
#     trimmed_data = trim_image_ids(data)

#     # Save the trimmed data to a new .pkl file
#     save_pkl(trimmed_data, output_file)

#     print(f"Trimmed data saved to {output_file}")
# except Exception as e:
#     print(f"An error occurred: {e}")




# import pickle

# def load_pkl(file_path):
#     """Load data from a pickle file."""
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)

# def save_pkl(data, file_path):
#     """Save data to a pickle file."""
#     with open(file_path, 'wb') as f:
#         pickle.dump(data, f)

# def trim_to_first_12000(data):
#     """Trim the dataset to only include the first 12,000 unique image IDs."""
#     unique_image_ids = set()
#     trimmed_data = []

#     for entry in data:
#         image_id = entry.get('image_id')
#         if image_id and len(unique_image_ids) < 12000:
#             if image_id not in unique_image_ids:
#                 unique_image_ids.add(image_id)
#             trimmed_data.append(entry)

#         # Stop when we reach 12,000 unique image IDs
#         if len(unique_image_ids) == 12000:
#             break

#     return trimmed_data

# # File paths
# input_file = 'vg_test_processed.pkl'  # Adjust this path if needed
# output_file = 'vg_test_.pkl'

# try:
#     # Load the dataset
#     data = load_pkl(input_file)

#     # Trim to the first 12,000 unique image IDs
#     trimmed_data = trim_to_first_12000(data)

#     # Save the trimmed dataset
#     save_pkl(trimmed_data, output_file)

#     print(f"Dataset trimmed to first 12,000 image IDs and saved to {output_file}")
# except Exception as e:
#     print(f"An error occurred: {e}")





# import pickle

# def load_pkl(file_path):
#     """Load data from a pickle file."""
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)

# def save_pkl(data, file_path):
#     """Save data to a pickle file."""
#     with open(file_path, 'wb') as f:
#         pickle.dump(data, f)

# def sort_by_image_id(data):
#     """Sort the dataset by image ID in ascending order."""
#     return sorted(data, key=lambda x: x.get('image_id', ''))

# # File paths
# input_file = 'vg_train_12000.pkl'  # Adjust this path if needed
# output_file = 'vg_test_sorted.pkl'

# try:
#     # Load the dataset
#     data = load_pkl(input_file)

#     # Sort the dataset by image ID
#     sorted_data = sort_by_image_id(data)

#     # Save the sorted dataset
#     save_pkl(sorted_data, output_file)

#     print(f"Dataset sorted by image ID and saved to {output_file}")
# except Exception as e:
#     print(f"An error occurred: {e}")


# import pickle
# import json

# # Paths to the input files
# pkl_file_path = "vg_train_12000.pkl"  # Replace with your actual .pkl file path
# json_file_path = "filtered_attri.json"  # Replace with your actual .json file path
# filtered_pkl_path = "filtered_vg_train.pkl"  # Output .pkl file with filtered data
# log_file_path = "remov_ids.log"  # Log file to store removed image IDs

# # Load the JSON file
# with open(json_file_path, 'r') as json_file:
#     json_data = json.load(json_file)

# # Extract the list of image IDs from the JSON file
# json_image_ids = {item['image_id'] for item in json_data}

# # Load the .pkl file
# with open(pkl_file_path, 'rb') as pkl_file:
#     pkl_data = pickle.load(pkl_file)

# # Filter the pkl data and track removed IDs
# filtered_data = []
# removed_ids = []

# for entry in pkl_data:
#     if entry['image_id'] in json_image_ids:
#         filtered_data.append(entry)
#     else:
#         removed_ids.append(entry['image_id'])

# # Save the filtered data to a new .pkl file
# with open(filtered_pkl_path, 'wb') as output_pkl:
#     pickle.dump(filtered_data, output_pkl)

# # Write the removed image IDs to a log file
# with open(log_file_path, 'w') as log_file:
#     for image_id in removed_ids:
#         log_file.write(f"Removed image_id: {image_id}\n")

# print(f"Filtered .pkl file saved to: {filtered_pkl_path}")
# print(f"Log file of removed IDs saved to: {log_file_path}")

# import pickle
# import random

# # Load the .pkl file
# pkl_file_path = "D:/WORK/PG/Project/vqa_project/data/processed/og/vqa_train_processed.pkl"  # Replace with your actual file path
# output_pkl_path = "vqa_train.pkl"  # Output file

# import pickle

# # Load the pickle file
# def load_pkl(file_path):
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)
#     return data

# # Extract data for the given image IDs
# def extract_data(pkl_file, output_file, num_images=10000):
#     data = load_pkl(pkl_file)
    
#     # Ensure the file contains a list of dictionaries
#     if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
#         raise ValueError("Pickle file should contain a list of dictionaries.")

#     # Extract first 2000 image entries
#     extracted_data = data[:num_images]

#     # Save extracted data to a new pickle file
#     with open(output_file, "wb") as f:
#         pickle.dump(extracted_data, f)
    
#     print(f"Extracted {len(extracted_data)} image entries and saved to {output_file}")

# # Usage
# pkl_file = "D:/WORK/PG/Project/vqa_project/data/processed/og/vqa_train_processed.pkl"  # Replace with your actual file path
# output_file = "vgtrain.pkl"

# extract_data(pkl_file, output_file)

# import pickle
# import random

# # Load the pickle file
# def load_pkl(file_path):
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)
#     return data

# # Extract and save two random sets of 1,000 samples each
# def extract_random_samples(pkl_file, output_file1, output_file2, num_samples=500):
#     data = load_pkl(pkl_file)
    
#     # Ensure the file contains a list of dictionaries
#     if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
#         raise ValueError("Pickle file should contain a list of dictionaries.")
    
#     # Shuffle the data randomly
#     random.shuffle(data)

#     # Extract two sets of 1,000 samples each
#     extracted_data1 = data[:num_samples]
#     extracted_data2 = data[num_samples:num_samples * 2]  # Next 1,000 samples

#     # Save the extracted data to new pickle files
#     with open(output_file1, "wb") as f:
#         pickle.dump(extracted_data1, f)

#     with open(output_file2, "wb") as f:
#         pickle.dump(extracted_data2, f)
    
#     print(f"Extracted {len(extracted_data1)} samples and saved to {output_file1}")
#     print(f"Extracted {len(extracted_data2)} samples and saved to {output_file2}")

# # Usage
# pkl_file = "vqa_train.pkl"  # Replace with your actual file path
# output_file1 = "sample5_1.pkl"
# output_file2 = "sample5_2.pkl"

# extract_random_samples(pkl_file, output_file1, output_file2)

import pickle

# Load the pickle file
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

# Extract data for the next 10000 images (from 10000 to 19999)
def extract_data(pkl_file, output_file, start=1001, end=2001):
    data = load_pkl(pkl_file)
    
    # Ensure the file contains a list of dictionaries
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("Pickle file should contain a list of dictionaries.")
    
    # Extract the desired range of image entries
    extracted_data = data[start:end]

    # Save extracted data to a new pickle file
    with open(output_file, "wb") as f:
        pickle.dump(extracted_data, f)

    print(f"Extracted {len(extracted_data)} image entries (from {start} to {end-1}) and saved to {output_file}")

# Usage
pkl_file = "D:/WORK/PG/Project/vqa_project/data/processed/og/vqa_val_processed.pkl"
output_file = "vgval.pkl"

extract_data(pkl_file, output_file)
