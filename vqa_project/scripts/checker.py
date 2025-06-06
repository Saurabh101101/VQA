# import json
# import math

# def check_json_file(file_path):
#     try:
#         # Load JSON file
#         with open(file_path, 'r') as file:
#             data = json.load(file)

#         if not isinstance(data, list):
#             print("The JSON file does not contain a list at the root. Exiting.")
#             return

#         malformed_entries = []
#         empty_entries = []
#         nan_entries = []

#         # Iterate over each entry in the JSON file
#         for i, entry in enumerate(data):
#             if not isinstance(entry, dict):
#                 malformed_entries.append((i, entry))
#             elif not entry:
#                 empty_entries.append((i, entry))
#             else:
#                 # Check for NaN values in the entry
#                 for key, value in entry.items():
#                     if isinstance(value, float) and math.isnan(value):
#                         nan_entries.append((i, key, value))

#         # Print malformed entries
#         if malformed_entries:
#             print("Malformed entries found:")
#             for index, entry in malformed_entries:
#                 print(f"Index {index}: {entry}")
#         else:
#             print("No malformed entries found.")

#         # Print empty entries
#         if empty_entries:
#             print("Empty entries found:")
#             for index, entry in empty_entries:
#                 print(f"Index {index}: {entry}")
#         else:
#             print("No empty entries found.")

#         # Print NaN entries
#         if nan_entries:
#             print("Entries with NaN values found:")
#             for index, key, value in nan_entries:
#                 print(f"Index {index}, Key '{key}': {value}")
#         else:
#             print("No NaN values found.")

#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#     except FileNotFoundError:
#         print("The specified JSON file was not found.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# # Example usage
# # Replace 'your_file.json' with the path to your JSON file
# check_json_file('filtered_relationships.json')



# # import pickle
# # import math

# # def check_pkl_file(file_path):
# #     try:
# #         # Load pickle file
# #         with open(file_path, 'rb') as file:
# #             data = pickle.load(file)

# #         if not isinstance(data, list):
# #             print("The pickle file does not contain a list at the root. Exiting.")
# #             return

# #         malformed_entries = []
# #         empty_entries = []
# #         nan_entries = []

# #         # Iterate over each entry in the pickle file
# #         for i, entry in enumerate(data):
# #             if not isinstance(entry, dict):
# #                 malformed_entries.append((i, entry))
# #             elif not entry:
# #                 empty_entries.append((i, entry))
# #             else:
# #                 # Check for NaN values in the entry
# #                 for key, value in entry.items():
# #                     if isinstance(value, float) and math.isnan(value):
# #                         nan_entries.append((i, key, value))

# #         # Print malformed entries
# #         if malformed_entries:
# #             print("Malformed entries found:")
# #             for index, entry in malformed_entries:
# #                 print(f"Index {index}: {entry}")
# #         else:
# #             print("No malformed entries found.")

# #         # Print empty entries
# #         if empty_entries:
# #             print("Empty entries found:")
# #             for index, entry in empty_entries:
# #                 print(f"Index {index}: {entry}")
# #         else:
# #             print("No empty entries found.")

# #         # Print NaN entries
# #         if nan_entries:
# #             print("Entries with NaN values found:")
# #             for index, key, value in nan_entries:
# #                 print(f"Index {index}, Key '{key}': {value}")
# #         else:
# #             print("No NaN values found.")

# #     except pickle.UnpicklingError as e:
# #         print(f"Error unpickling data: {e}")
# #     except FileNotFoundError:
# #         print("The specified pickle file was not found.")
# #     except Exception as e:
# #         print(f"An unexpected error occurred: {e}")

# # # Example usage
# # # Replace 'your_file.pkl' with the path to your pickle file
# # check_pkl_file('vg_train_12000.pkl')

# import pickle
# from collections import Counter

# # Load the .pkl file
# def load_pkl(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# # Function to analyze image IDs
# def analyze_image_ids(file_path, key='image_id'):
#     data = load_pkl(file_path)
    
#     # Extract image IDs (assuming they are stored in a dictionary or list of dictionaries)
#     if isinstance(data, dict):
#         image_ids = data.get(key, [])  # Modify if necessary
#     elif isinstance(data, list):
#         image_ids = [item[key] for item in data if key in item]
#     else:
#         raise ValueError("Unsupported data format in the pickle file")
    
#     total_ids = len(image_ids)
#     unique_ids = len(set(image_ids))
#     id_counts = Counter(image_ids)
#     duplicates = {img_id: count for img_id, count in id_counts.items() if count > 1}
    
#     print(f"Total number of image IDs: {total_ids}")
#     print(f"Total number of unique image IDs: {unique_ids}")
    # print("Number of duplicates per image ID:")
    # for img_id, count in duplicates.items():
    #     print(f"Image ID {img_id}: {count} times")

# Example usage
# file_path = 'vg_val.pkl'  # Replace with the actual path
# analyze_image_ids(file_path)
import torch
state_dict = torch.load("D:/WORK/PG/Project/vqa_final/vqa_model.pth", map_location="cpu")
print(state_dict.keys())