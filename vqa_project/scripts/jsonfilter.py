# import json
# import pickle

# # Load the data from the .pkl file
# with open('vg_test_6000.pkl', 'rb') as pkl_file:
#     pkl_data = pickle.load(pkl_file)

# # Extract the image IDs from the .pkl data (assuming they are stored in a specific field)
# pkl_image_ids = set()
# for item in pkl_data:
#     if 'image_id' in item:  # Replace 'image_id' with the actual key used for image IDs in your pkl data
#         pkl_image_ids.add(item['image_id'])

# # Load the JSON file
# with open('D:/WORK/PG/Project/vqa_project/data/processed/relationships.json', 'r') as json_file:
#     json_data = json.load(json_file)

# # Filter the JSON data based on the image IDs in the .pkl file
# filtered_json_data = [item for item in json_data if item.get('image_id') in pkl_image_ids]

# # Save the filtered data to a new JSON file in a compact format
# with open('newfiltered_relationships6.json', 'w') as output_file:
#     json.dump(filtered_json_data, output_file, indent=4)

# print("Filtered JSON data has been saved to 'filtered_relationships_train.json'.")



# import json

# def combine_json_files(file1, file2, output_file):
#     try:
#         # Load the JSON files
#         with open(file1, 'r') as f1, open(file2, 'r') as f2:
#             data1 = json.load(f1)
#             data2 = json.load(f2)
        
#         # Combine data while preserving duplicates
#         combined_data = data1 + data2
        
#         # Save the result to the output file
#         with open(output_file, 'w') as out:
#             json.dump(combined_data, out, indent=4)
        
#         print(f"Combined file saved to {output_file}.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Example usage
# file1 = 'newfiltered_relationships12.json'
# file2 = 'newfiltered_relationships6.json'
# output_file = 'relationships_filtered.json'

# combine_json_files(file1, file2, output_file)

# import json

# # Paths to your JSON file and the log file
# input_file = "relationships_filtered.json"
# # output_file = "filtered_relation.json"
# log_file = "removed_ids.log"

# # Load the JSON file
# with open(input_file, 'r') as f:
#     data = json.load(f)

# # Check if the data is a list
# if not isinstance(data, list):
#     raise ValueError("The JSON structure is not a list of objects.")

# # Filter entries and log removed IDs
# filtered_data = []
# removed_ids = []

# for entry in data:
#     if "relationships" in entry and not entry["relationships"]:
#         removed_ids.append(entry.get("image_id", "Unknown ID"))
#     else:
#         filtered_data.append(entry)

# # Save the filtered data
# # with open(output_file, 'w') as f:
# #     json.dump(filtered_data, f, indent=4)

# # Save the log file
# with open(log_file, 'w') as f:
#     for image_id in removed_ids:
#         f.write(f"Removed image_id: {image_id}\n")

# # print(f"Filtered data saved to {output_file}")
# print(f"Log of removed image IDs saved to {log_file}")


# import json

# # Function to parse the log file and extract removed image IDs
# def get_removed_image_ids(log_file_path):
#     removed_ids = set()
#     with open(log_file_path, 'r') as log_file:
#         for line in log_file:
#             if "Removed image_id:" in line:
#                 try:
#                     image_id = int(line.split(":")[-1].strip())
#                     removed_ids.add(image_id)
#                 except ValueError:
#                     print(f"Skipping invalid line: {line.strip()}")
#     return removed_ids

# # Function to remove entries with specific image IDs from a JSON file
# def remove_image_ids_from_json(json_file_path, output_file_path, removed_ids):
#     with open(json_file_path, 'r') as json_file:
#         data = json.load(json_file)

#     # Assuming the JSON is a list of dictionaries
#     filtered_data = [entry for entry in data if entry.get('image_id') not in removed_ids]

#     with open(output_file_path, 'w') as output_file:
#         json.dump(filtered_data, output_file, indent=4)

#     print(f"Filtered JSON written to {output_file_path}")

# # Paths to the log file and JSON file
# log_file_path = 'removed_ids.log'  # Replace with your log file path
# json_file_path = 'attributes_filtered.json'       # Replace with your JSON file path
# output_file_path = 'filtered_attri.json'  # Output JSON file

# # Process the log file and remove the entries from the JSON
# removed_ids = get_removed_image_ids(log_file_path)
# remove_image_ids_from_json(json_file_path, output_file_path, removed_ids)

# import json

# # Paths to your JSON file and the log file
# input_file = "relationships_filtered.json"
# # output_file = "filtered_output.json"
# log_file = "removed_ids.log"

# # Load the JSON file
# with open(input_file, 'r') as f:
#     data = json.load(f)

# # Check if the data is a list
# if not isinstance(data, list):
#     raise ValueError("The JSON structure is not a list of objects.")

# # Filter entries and log removed IDs
# filtered_data = []
# removed_ids = []

# for entry in data:
#     if "relationships" in entry and not entry["relationships"]:
#         removed_ids.append(entry.get("image_id", "Unknown ID"))
#     else:
#         filtered_data.append(entry)

# # Save the filtered data
# # with open(output_file, 'w') as f:
#     # json.dump(filtered_data, f, indent=4)

# # Save the log file
# with open(log_file, 'w') as f:
#     for image_id in removed_ids:
#         f.write(f"Removed image_id: {image_id}\n")

# # print(f"Filtered data saved to {output_file}")
# print(f"Log of removed image IDs saved to {log_file}")

# import json

# # Function to remove objects with empty synsets
# def remove_empty_synsets(input_file, output_file):
#     # Load the JSON data from the input file
#     with open(input_file, 'r') as file:
#         data = json.load(file)

#     # Filter out the objects where 'synsets' is an empty list
#     for item in data:
#         item['attributes'] = [attribute for attribute in item['attributes'] if attribute['synsets']]

#     # Save the updated data to the output file
#     with open(output_file, 'w') as file:
#         json.dump(data, file, indent=4)

# # Example usage
# input_file = 'filtered_attri.json'  # Replace with your input file path
# output_file = 'filtered_attrib.json'  # Replace with your output file path
# remove_empty_synsets(input_file, output_file)

# import json

# # Function to remove objects with empty attributes
# def remove_empty_attributes(input_file, output_file):
#     # Load the JSON data from the input file
#     with open(input_file, 'r') as file:
#         data = json.load(file)

#     # Filter out the objects where 'attributes' is an empty list
#     data = [item for item in data if item['attributes']]

#     # Save the updated data to the output file
#     with open(output_file, 'w') as file:
#         json.dump(data, file, indent=4)

# # Example usage
# input_file = 'filtered_attrib.json'  # Replace with your input file path
# output_file = 'filtered_attribu.json'  # Replace with your output file path
# remove_empty_attributes(input_file, output_file)

# import json

# # Recursive function to remove empty synsets
# def remove_empty_synsets_recursive(data):
#     if isinstance(data, list):
#         # If it's a list, recursively process each element
#         return [remove_empty_synsets_recursive(item) for item in data if not (isinstance(item, dict) and "synsets" in item and not item["synsets"])]
#     elif isinstance(data, dict):
#         # If it's a dictionary, process its keys and values
#         return {key: remove_empty_synsets_recursive(value) for key, value in data.items() if not (key == "synsets" and value == [])}
#     else:
#         # If it's neither, return the data as is
#         return data

# # Function to load, clean, and save the JSON
# def clean_json(input_file, output_file):
#     # Load the JSON data from the input file
#     with open(input_file, 'r') as file:
#         data = json.load(file)

#     # Clean the data
#     cleaned_data = remove_empty_synsets_recursive(data)

#     # Save the cleaned data to the output file
#     with open(output_file, 'w') as file:
#         json.dump(cleaned_data, file, indent=4)

# # Example usage
# input_file = 'filtered_relation.json'  # Replace with your input file path
# output_file = 'filtered_relationsh.json'  # Replace with your output file path
# clean_json(input_file, output_file)

# import json

# # Function to remove objects with empty relationships
# def remove_empty_relationships(input_file, output_file):
#     # Load the JSON data from the input file
#     with open(input_file, 'r') as file:
#         data = json.load(file)

#     # Filter out the objects where 'relationships' is an empty list
#     data = [item for item in data if not (isinstance(item, dict) and "relationships" in item and not item["relationships"])]

#     # Save the updated data to the output file
#     with open(output_file, 'w') as file:
#         json.dump(data, file, indent=4)

# # Example usage
# input_file = 'filtered_relationsh.json'  # Replace with your input file path
# output_file = 'filtered_relationshio.json'  # Replace with your output file path
# remove_empty_relationships(input_file, output_file)





# import json

# def filter_common_image_ids(file1_path, file2_path, output1_path, output2_path, log_file):
#     # Load JSON data from the files
#     with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
#         data1 = json.load(file1)
#         data2 = json.load(file2)
    
#     # Extract image IDs from both files
#     ids1 = {item['image_id'] for item in data1}
#     ids2 = {item['image_id'] for item in data2}
    
#     # Find common and uncommon image IDs
#     common_ids = ids1 & ids2
#     uncommon_ids = ids1 ^ ids2  # Symmetric difference
    
#     # Log uncommon image IDs
#     with open(log_file, 'w') as log:
#         log.write("Uncommon Image IDs:\n")
#         for img_id in sorted(uncommon_ids):
#             log.write(f"{img_id}\n")
    
#     # Filter data with common image IDs
#     filtered_data1 = [item for item in data1 if item['image_id'] in common_ids]
#     filtered_data2 = [item for item in data2 if item['image_id'] in common_ids]
    
#     # Save filtered data to new JSON files
#     with open(output1_path, 'w') as out1, open(output2_path, 'w') as out2:
#         json.dump(filtered_data1, out1, indent=4)
#         json.dump(filtered_data2, out2, indent=4)

#     print(f"Filtered JSON files saved to {output1_path} and {output2_path}.")
#     print(f"Log of uncommon image IDs saved to {log_file}.")

# # Example usage
# filter_common_image_ids(
#     file1_path='cleaned_relationships.json',
#     file2_path='cleaned_attributes.json',
#     output1_path='relationshionFil.json',
#     output2_path='attributesFil.json',
#     log_file='uncommon_ids.log'
# )

# import pickle

# def remove_image_ids_from_pkl(pkl1_path, pkl2_path, log_file, output1_path, output2_path):
#     # Load the log file with IDs to remove
#     with open(log_file, 'r') as log:
#         ids_to_remove = {line.strip() for line in log if line.strip()}
    
#     # Function to filter data from a pkl file
#     def filter_pkl_file(pkl_path, ids_to_remove):
#         with open(pkl_path, 'rb') as file:
#             data = pickle.load(file)
#         # Assume the data is a list of dictionaries with an "image_id" key
#         filtered_data = [item for item in data if item.get('image_id') not in ids_to_remove]
#         return filtered_data

#     # Filter both pkl files
#     filtered_data1 = filter_pkl_file(pkl1_path, ids_to_remove)
#     filtered_data2 = filter_pkl_file(pkl2_path, ids_to_remove)

#     # Save the filtered data to new pkl files
#     with open(output1_path, 'wb') as out1, open(output2_path, 'wb') as out2:
#         pickle.dump(filtered_data1, out1)
#         pickle.dump(filtered_data2, out2)

#     print(f"Filtered PKL files saved to {output1_path} and {output2_path}.")

# # Example usage
# remove_image_ids_from_pkl(
#     pkl1_path='f_vg_train.pkl',
#     pkl2_path='f_vg_test.pkl',
#     log_file='uncommon_ids.log',
#     output1_path='fil_vg_train.pkl',
#     output2_path='fil_vg_test.pkl'
# )

import pickle
import json

def filter_pkl_by_json(json_path, pkl_path, output_pkl_path):
    # Load image IDs from JSON file
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
        # Extract image IDs (assume the JSON is a list of dictionaries with 'image_id' key)
        valid_ids = {item['image_id'] for item in json_data}
    
    # Load data from the PKL file
    with open(pkl_path, 'rb') as pkl_file:
        pkl_data = pickle.load(pkl_file)
    
    # Filter PKL data to include only entries with valid image IDs
    filtered_data = [item for item in pkl_data if item.get('image_id') in valid_ids]
    
    # Save the filtered data to a new PKL file
    with open(output_pkl_path, 'wb') as output_file:
        pickle.dump(filtered_data, output_file)
    
    print(f"Filtered PKL file saved to {output_pkl_path}.")

# Example usage
filter_pkl_by_json(
    json_path='cleaned_relationships.json',
    pkl_path='f_vg_train.pkl',
    output_pkl_path='final_vg_train.pkl'
)

# import json

# def filter_invalid_relationships(relationships_file, attributes_file, output_file):
#     # Load JSON files
#     with open(relationships_file, "r") as rel_file:
#         relationships_data = json.load(rel_file)
    
#     with open(attributes_file, "r") as attr_file:
#         attributes_data = json.load(attr_file)
    
#     # Function to get all object_ids from attributes
#     def extract_object_ids(attributes):
#         return {attr["object_id"] for attr in attributes.get("attributes", [])}
    
#     # Filter entries with valid object_ids in relationships
#     cleaned_data = []
#     for entry in relationships_data:
#         attr_entry = next((attr for attr in attributes_data if attr["image_id"] == entry["image_id"]), None)
        
#         if not attr_entry:
#             # Skip if there's no corresponding attributes entry
#             continue
        
#         # Get valid object_ids from attributes
#         valid_object_ids = extract_object_ids(attr_entry)
        
#         # Check relationships
#         valid_relationships = []
#         for rel in entry.get("relationships", []):
#             obj_id = rel["object"]["object_id"]
#             subj_id = rel["subject"]["object_id"]
            
#             # Add relationship only if both object_id and subject_id are valid
#             if obj_id in valid_object_ids and subj_id in valid_object_ids:
#                 valid_relationships.append(rel)
        
#         # Add the entry only if valid relationships remain
#         if valid_relationships:
#             entry["relationships"] = valid_relationships
#             cleaned_data.append(entry)
    
#     # Save cleaned relationships data
#     with open(output_file, "w") as out_file:
#         json.dump(cleaned_data, out_file, indent=4)
    
#     print(f"Original relationships size: {len(relationships_data)}")
#     print(f"Cleaned relationships size: {len(cleaned_data)}")
#     print(f"Cleaned data saved to {output_file}")


# # Example usage
# relationships_file = "relationshionF.json"  # Input file with relationships
# attributes_file = "attributesF.json"        # Input file with attributes
# output_file = "cleaned_relationships.json" # Output file for cleaned data

# filter_invalid_relationships(relationships_file, attributes_file, output_file)


# Example usage
# relationships_file = "relationshionF.json"         # Input file with relationships
# attributes_file = "attributesF.json"               # Input file with attributes
# output_relationships_file = "cleaned_relationships.json"  # Output file for cleaned relationships
# output_attributes_file = "cleaned_attributes.json"        # Output file for cleaned attributes

# import json

# def filter_attributes_by_relationships(cleaned_relationships_file, attributes_file, output_file):
#     # Load cleaned relationships and attributes files
#     with open(cleaned_relationships_file, "r") as rel_file:
#         cleaned_relationships_data = json.load(rel_file)
    
#     with open(attributes_file, "r") as attr_file:
#         attributes_data = json.load(attr_file)
    
#     # Extract valid object_ids from cleaned relationships
#     valid_object_ids = set()
#     for entry in cleaned_relationships_data:
#         for rel in entry.get("relationships", []):
#             valid_object_ids.add(rel["object"]["object_id"])
#             valid_object_ids.add(rel["subject"]["object_id"])
    
#     # Filter attributes entries based on valid object_ids
#     cleaned_attributes_data = []
#     for entry in attributes_data:
#         # Filter attributes array within each image entry
#         filtered_attributes = [
#             attr for attr in entry.get("attributes", [])
#             if attr["object_id"] in valid_object_ids
#         ]
        
#         if filtered_attributes:
#             entry["attributes"] = filtered_attributes
#             cleaned_attributes_data.append(entry)
    
#     # Save cleaned attributes data
#     with open(output_file, "w") as out_file:
#         json.dump(cleaned_attributes_data, out_file, indent=4)
    
#     print(f"Original attributes size: {len(attributes_data)}")
#     print(f"Cleaned attributes size: {len(cleaned_attributes_data)}")
#     print(f"Cleaned attributes saved to {output_file}")


# # Example usage
# cleaned_relationships_file = "cleaned_relationships.json"  # Input file with cleaned relationships
# attributes_file = "attributesF.json"                       # Input file with original attributes
# output_file = "cleaned_attributes.json"                   # Output file for cleaned attributes

# filter_attributes_by_relationships(cleaned_relationships_file, attributes_file, output_file)
