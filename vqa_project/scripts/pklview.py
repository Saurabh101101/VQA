# # # import pickle

# # # file_path = 'D:/WORK/PG/Project/vqa_project/data/raw/processed/tokenized_text.pkl'

# # # try:
# # #     with open(file_path, 'rb') as file:
# # #         data = pickle.load(file)
# # #     print(data)
# # # except FileNotFoundError:
# # #     print(f"File not found: {file_path}")
# # # except pickle.UnpicklingError:
# # #     print("Error: The file content is not a valid pickle format.")
# # # except EOFError:
# # #     print("Error: The file is incomplete or corrupted.")
# # # except Exception as e:
# # #     print(f"An unexpected error occurred: {e}")

# # import pickle
# # # D:\WORK\PG\Project\vqa_project\data\raw\processed\image_features_filtered.pkl
# # image_features_filtered_path = "D:/WORK/PG/Project/vqa_project/data/raw/processed/encoded_features.pkl"

# # with open(image_features_filtered_path, "rb") as f:
# #     image_features_filtered = pickle.load(f)
# # print(type(image_features_filtered))
# # print(len(image_features_filtered))
# # print(encoded_features())  # Check the top-level keys
# # print("Values:", list(image_features_filtered.values()))

# import pickle

# vg_train_processed_path = 'D:/WORK/PG/Project/vqa_project/data/processed/og/new_vg_val_processed.pkl'

# with open(vg_train_processed_path, "rb") as f:
#     vg_train_processed = pickle.load(f)

# print(type(vg_train_processed))  # Should print `dict`
# print(vg_train_processed[:100])
# print("Keys in encoded_features:", vg_train_processed.keys())  # Should include "attributes" and "relationships"

# # # # Inspect attributes and relationships
# # # print("Type of attributes:", type(encoded_features["attributes"]))
# # # print("Type of relationships:", type(encoded_features["relationships"]))

# # # # Print a sample of attributes
# # # print("Sample of attributes:", encoded_features["attributes"][:5])  # Adjust indexing if needed

# # with open("D:/WORK/PG/Project/combined_data.pkl", "rb") as f:
# # # with open("D:/WORK/PG/Project/vqa_project/data/processed/combined_data.pkl", "rb") as f:
# #     combined_data = pickle.load(f)
# #     print(type(combined_data))  # Should print <class 'list'>
# #     print(len(combined_data)) 
# #     # print(combined_data.keys())
# #     print(list(combined_data.items())[:2])  # Adjust the number 2 to how many items you want
# # # Print the total number of en

# # import pickle

# # # Load all processed files
# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/image_features.pkl", "rb") as f:
# #     image_features = pickle.load(f)

# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/encoded_features.pkl", "rb") as f:
# #     encoded_features = pickle.load(f)

# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/tokenized_text.pkl", "rb") as f:
# #     tokenized_text = pickle.load(f)
# # print(f"Number of image features: {len(image_features)}")
# # print(f"Number of aligned attributes: {len(encoded_features)}")
# # print(f"Number of aligned relationships: {len(encoded_features['relationships'])}")
# # # Extract image IDs from all sources
# # # image_feature_ids = set(image_features.keys())
# # # encoded_feature_ids = set(encoded_features["attributes"].keys())
# # # tokenized_text_ids = set(tokenized_text.keys())

# # # # Check alignment
# # # assert image_feature_ids == encoded_feature_ids == tokenized_text_ids, "Data mismatch!"
# # # print("All datasets are perfectly aligned!")


# # import pickle

# # # Load data
# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/image_features.pkl", "rb") as f:
# #     all_image_features = pickle.load(f)

# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/combined_data.pkl", "rb") as f:
# #     combined_data = pickle.load(f)

# # # Get the image filenames from combined_data
# # image_ids = set(entry["image_id"] for entry in combined_data)

# # # Filter image features
# # filtered_image_features = {img_id: all_image_features[img_id] for img_id in image_ids if img_id in all_image_features}

# # # Save the filtered features
# # # with open("image_features_filtered.pkl", "wb") as f:
# # #     pickle.dump(filtered_image_features, f)

# # print(f"Filtered image features: {len(filtered_image_features)}")

# # import pickle

# # Load the combined_data.pkl file
# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/tokenized_text.pkl", "rb") as f:
# #     tokenized_text = pickle.load(f)
# # print(type(tokenized_text))
# # print(len(tokenized_text))
# # print("Keys in tokenized_text:", tokenized_text.keys())
# # # d = {'a': 'apple', 'b': 'banana', 'c': 'cherry'}  
# # lengths = [len(v) for v in tokenized_text.values()]
# # print(lengths)

# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/image_features_filtered.pkl", "rb") as f:
# #     image_features_filtered = pickle.load(f)
# # print(type(image_features_filtered))
# # print(len(image_features_filtered))
# # print("Keys in image_features_filtered:", image_features_filtered.keys())
# # # d = {'a': 'apple', 'b': 'banana', 'c': 'cherry'}  
# # lengths = [len(v) for v in image_features_filtered.values()]
# # print(lengths)

# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/encoded_features.pkl", "rb") as f:
# #     encoded_features = pickle.load(f)
# # print(type(encoded_features))
# # print(len(encoded_features))
# # print("Keys in encoded_features:", encoded_features.keys())
# # # d = {'a': 'apple', 'b': 'banana', 'c': 'cherry'}  
# # lengths = [len(v) for v in encoded_features.values()]
# # print(lengths)
# # for key, value in list(encoded_features.items())[:3]:  # Print first 3 items
# #     print(f"Key: {key}, Value: {value}")

# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/combined_data.pkl", 'rb') as f:
# #     combined_data = pickle.load(f)

# # # Print a sample of combined_data
# # for key, value in list(combined_data.items())[:3]:  # Print first 3 items
# #     print(f"Key: {key}, Value: {value}")


# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/combined_data.pkl", "rb") as f:
# #     combined_data = pickle.load(f)
# # print(type(combined_data))
# # print(len(combined_data))
# # print("Keys in combined_data:", combined_data.keys())
# # # d = {'a': 'apple', 'b': 'banana', 'c': 'cherry'}  
# # lengths = [len(v) for v in combined_data.values()]
# # print(lengths)

# # print(f"Total images: {len(combined_data)}")  # Should be 848
# # total_entries = sum(len(entries) for entries in combined_data.values())
# # print(f"Total question-answer pairs: {total_entries}")  # Should match 4690

# # print(combined_data[:5]) 
# # Inspect the type and sample entries
# # print(f"Type of combined_data: {type(combined_data)}")
# # print(f"Number of entries: {len(combined_data)}")
# # print("Sample entry structure:")
# # print(combined_data[0])
# # print(combined_data[1])# Print the first entry

# import pickle

# # Load image features
# with open("D:/WORK/PG/Project/vqa_project/data/processed/vg_image_features_test.pkl", "rb") as f:
#     all_image_features = pickle.load(f)

# # Inspect type and structure
# print(f"Type of all_image_features: {type(all_image_features)}")
# print(len(all_image_features))
# for key in list(all_image_features.keys())[:5]:  # Print first 5 keys
#     print(f"Key: {key}, Value Type: {(all_image_features[key])}")

# # import pickle

# # # Load combined data
# # with open("D:/WORK/PG/Project/vqa_project/data/raw/processed/combined_data.pkl", "rb") as f:
# #     combined_data = pickle.load(f)


# # Check the type of the first entry
# # first_key = next(iter(combined_data))  # Dynamically get the first key
# # print(f"Type of first entry: {type(combined_data[first_key])}")
# # # Access the first entry
# # first_key = list(combined_data.keys())[0]
# # first_entry = combined_data[first_key]

# # # Check the type of the first element in the list
# # type_of_first_element = type(first_entry[0])
# # print(f"Type of first element: {type_of_first_element}")
# # # Inspect the keys of the first dictionary
# # print(f"Keys in first element: {first_entry[0].keys()}")

# # import pickle
# # import os

# # processed_dir = "D:/WORK/PG/Project/vqa_project/data/raw/processed"
# # image_features_filtered_path = os.path.join(processed_dir, "image_features_filtered.pkl")
# # encoded_features_path = os.path.join(processed_dir, "encoded_features.pkl")

# # # Load files
# # with open(image_features_filtered_path, "rb") as f:
# #     image_features_filtered = pickle.load(f)
# #     print(type(image_features_filtered))
# #     print("Number of image features:", len(image_features_filtered))
# #     print("Sample image feature keys:", list(image_features_filtered.keys())[:5])

# # with open(encoded_features_path, "rb") as f:
# #     encoded_features = pickle.load(f)
# #     print(type(encoded_features))
# #     print("Number of attributes:", len(encoded_features.get("attributes", [])))
# #     print("Number of relationships:", len(encoded_features.get("relationships", [])))
# #     print("Sample attributes:", encoded_features["attributes"][:3])
# #     print("Sample relationships:", encoded_features["relationships"][:3])

# import pickle

# Path to the pkl file
# file_path = "D:/WORK/PG/Project/vqa_project/data/processed/vg_test_processed.pkl"

# # Load the data
# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# print(len(data))
# # Inspect the first sample
# if isinstance(data, list):
#     first_sample = data[0]
#     print("Keys in the first sample:", first_sample.keys())
#     print("Shape of image_features:", first_sample['image_features'].shape)
#     print("Answers example:", first_sample['answer'])
#     print("Answers example:", first_sample['image_id'])
# else:
#     print("Unexpected data format:", type(data))
# if isinstance(data, list):
#     first_sample = data[1]
#     print("Keys in the first sample:", first_sample.keys())
#     print("Shape of image_features:", first_sample['image_features'].shape)
#     print("Answers example:", first_sample['answer'])
#     print("Answers example:", first_sample['image_id'])
# else:
#     print("Unexpected data format:", type(data))

# import pickle

# def get_dataset_size(pkl_file):
#     with open(pkl_file, 'rb') as f:
#         data = pickle.load(f)
#         if not data or len(data) < 1:
#             raise ValueError(f"The dataset in {pkl_file} is empty or malformed.")
#         print(f"Number of samples in the test dataset: {len(data)}")
#         return len(data)

# # Replace 'your_dataset.pkl' with the path to your dataset file
# dataset_path = "D:/WORK/PG/Project/vqa_project/data/processed/vqa_train_processed.pkl"
# get_dataset_size(dataset_path)

import pickle

# Load the pickle file
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to count entries and unique images
def analyze_pkl(file_path, image_key='image_id'):
    data = load_pkl(file_path)
    
    num_entries = len(data)  # Number of total entries
    unique_images = set()  # To store unique image IDs

    for entry in data:
        if image_key in entry:
            unique_images.add(entry[image_key])

    num_unique_images = len(unique_images)

    print(f"Number of Entries: {num_entries}")
    print(f"Total Unique Images: {num_unique_images}")

# Provide your .pkl file path
pkl_file_path = "vgval.pkl"
analyze_pkl(pkl_file_path)
