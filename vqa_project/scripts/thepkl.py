import pickle

# Define paths to the original and new files
# original_test_path ="D:/WORK/PG/Project/vqa_project/data/processed/vg_test_processed.pkl" # Replace with your test.pkl file path
# # original_train_path = "D:/WORK/PG/Project/vqa_project/data/processed/vg_train_processed.pkl" # Replace with your train.pkl file path
# new_test_path = "D:/WORK/PG/Project/vqa_project/data/processed/new_vg_test_processed.pkl"
# new_train_path = "new_train.pkl"

# Desired sample sizes
# desired_test_size = 349779
# # desired_train_size = 443757

# def sample_and_save_sequentially(input_path, output_path, sample_size):
#     # Load the original dataset
#     with open(input_path, "rb") as file:
#         data = pickle.load(file)

#     # Select the first 'sample_size' samples
#     sampled_data = data[:sample_size]

#     # Save the sampled dataset to a new file
#     with open(output_path, "wb") as file:
#         pickle.dump(sampled_data, file)

# Process the test dataset
# sample_and_save_sequentially(original_test_path, new_test_path, desired_test_size)

# Process the train dataset
# sample_and_save_sequentially(original_train_path, new_train_path, desired_train_size)

# print(f"New test dataset saved to {new_test_path} with {desired_test_size} samples.")
# print(f"New train dataset saved to {new_train_path} with {desired_train_size} samples.")

# def get_dataset_size(pkl_file):
#     with open(pkl_file, 'rb') as f:
#         data = pickle.load(f)
#         if not data or len(data) < 1:
#             raise ValueError(f"The dataset in {pkl_file} is empty or malformed.")
#         print(f"Number of samples in the dataset: {len(data)}")
#         return len(data)

# Replace 'your_dataset.pkl' with the path to your dataset file
# datasetrain_path = "D:/WORK/PG/Project/vqa_project/data/processed/vg_train_processed.pkl"
# datasetest_path = "D:/WORK/PG/Project/vqa_project/data/processed/vg_test_processed.pkl"

# get_dataset_size(datasetrain_path)
# get_dataset_size(new_train_path)
# print("___")
# get_dataset_size(new_test_path)
# get_dataset_size(datasetest_path)

# import pickle

# # Function to search for an image ID in a pickle file containing a list of dictionaries
# def search_image_id(pkl_file_path, image_id):
#     with open(pkl_file_path, 'rb') as f:
#         data = pickle.load(f)
    
#     # Loop through the list of dictionaries and search for the image ID
#     for entry in data:
#         if entry.get('image_id') == image_id:  # Adjust the key as needed based on your data structure
#             return entry
    
#     return f"Image ID {image_id} not found."

# # Example usage
# pkl_file_path = "D:/WORK/PG/Project/vqa_project/data/processed/vg_train_processed.pkl"  # Path to your pickle file
# image_id = 2345675  # The image ID you're searching for
# result = search_image_id(pkl_file_path, image_id)
# print(result)
import pickle

# Load the pickle file
with open("D:/WORK/PG/Project/vqa_project/data/processed/old/new_vg_test_processed.pkl", "rb") as f:  # Replace 'data.pkl' with your actual file name
    data = pickle.load(f)

# Modify the dictionary keys
for entry in data:
    entry["answers"] = entry.pop("answer")  # Rename 'answer' to 'answers'

# Save the modified pickle file
with open("new_vg_test_processed.pkl", "wb") as f:  # Saves as a new file to avoid overwriting
    pickle.dump(data, f)

print("Modification complete! Saved as 'modified_data.pkl'.")

# import pickle
# import os

# def combine_pkl_files(file_list, output_file):
#     combined_data = []

#     for file in file_list:
#         with open(file, "rb") as f:
#             data = pickle.load(f)
#             combined_data.extend(data)  # Combine data

#     with open(output_file, "wb") as f:
#         pickle.dump(combined_data, f)

#     print(f"Combined data saved as '{output_file}'.")

# # Example usage
# file_list = ["vg_train_processssed.pkl", "D:/WORK/PG/Project/vqa_project/data/processed/vqa_train_processed.pkl"]  # List your files here
# output_file = "combined_data.pkl"
# combine_pkl_files(file_list, output_file)

# import pickle

# def combine_first_100_entries(file1, file2, output_file):
#     combined_data = []

#     for file in [file1, file2]:
#         with open(file, "rb") as f:
#             data = pickle.load(f)
#             combined_data.extend(data[:100])  # Take the first 100 entries

#     with open(output_file, "wb") as f:
#         pickle.dump(combined_data, f)

#     print(f"Combined data saved as '{output_file}'.")

# # Example usage
# file1 = "vg_train_processssed.pkl"  # Replace with your actual file names
# file2 = "D:/WORK/PG/Project/vqa_project/data/processed/vqa_val_processed.pkl"
# output_file = "val_data.pkl"

# combine_first_100_entries(file1, file2, output_file)

import pickle

def combine_pkl_files(file1, file2, output_file):
    # Load the first file
    with open(file1, 'rb') as f:
        data1 = pickle.load(f)

    # Load the second file
    with open(file2, 'rb') as f:
        data2 = pickle.load(f)

    # Ensure both are lists
    if not isinstance(data1, list) or not isinstance(data2, list):
        raise ValueError("Both .pkl files must contain lists of dictionaries.")

    # Merge the lists
    combined_data = data1 + data2

    # Save to a new .pkl file
    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"Combined {len(data1)} + {len(data2)} items into {output_file}")

# Example usage
combine_pkl_files("vqa_val.pkl", "vg_val.pkl", "val.pkl")
