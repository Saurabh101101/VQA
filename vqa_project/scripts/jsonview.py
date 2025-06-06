import json

def extract_json_entries(input_file, output_file, num_entries):
    """
    Extract a subset of entries from a large JSON file.

    :param input_file: Path to the large JSON file.
    :param output_file: Path to save the extracted JSON data.
    :param num_entries: Number of entries to extract.
    """
    try:
        with open(input_file, 'r') as file:
            data = json.load(file)
        
        # Ensure the file contains a list or dictionary
        if isinstance(data, list):
            subset = data[:num_entries]
        elif isinstance(data, dict):
            subset = {k: data[k] for i, k in enumerate(data) if i < num_entries}
        else:
            print("Unsupported JSON structure. Please use a list or dictionary.")
            return
        
        with open(output_file, 'w') as file:
            json.dump(subset, file)
        
        print(f"Extracted {num_entries} entries to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = 'relationships_sample6.json'
output_file = 'relationships_sample30.json'
num_entries = 30000  # Adjust the number of entries you need
extract_json_entries(input_file, output_file, num_entries)


# # print(list(self.relationships.keys())[:5])
# # print(list(input_file.keys())[:5])
# # print(type(input_file))

# # with open('D:/WORK/PG/Project/vqa_project/data/processed/relationships.json', 'r') as f:
# #     relationships = json.load(f)
# #     print(any(item['image_id'] == 2415074 for item in relationships))  # Should return True

# # with open('D:/WORK/PG/Project/vqa_project/data/processed/attributes.json', 'r') as f:
# #     attributes = json.load(f)
# #     print(any(item['image_id'] == 2415074 for item in attributes))

# import pickle

# # Load the .pkl file
# with open("D:/WORK/PG/Project/vqa_project/data/processed/vg_train_processed.pkl", 'rb') as file:
#     data = pickle.load(file)

# # Extract image_ids from the data (assuming the data structure contains 'image_id' for each entry)
# image_ids = [entry['image_id'] for entry in data]

# # Get the total number of unique image_ids
# unique_image_ids = len((image_ids))

# # Print the total count of unique image_ids
# print(f"Total number of unique image IDs: {unique_image_ids}")
