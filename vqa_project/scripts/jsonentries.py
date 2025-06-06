# import json

# # Replace 'your_file.json' with the path to your JSON file
# file_path = 'D:/WORK/PG/Project/vqa_project/data/processed/relationships.json'

# try:
#     # Load the JSON data
#     with open(file_path, 'r') as file:
#         data = json.load(file)

#     # Recursive function to count 'image_id' occurrences
#     def count_image_ids(obj):
#         count = 0
#         if isinstance(obj, dict):
#             for key, value in obj.items():
#                 if key == 'image_id':
#                     count += 1
#                 count += count_image_ids(value)
#         elif isinstance(obj, list):
#             for item in obj:
#                 count += count_image_ids(item)
#         return count

#     # Count total 'image_id'
#     total_image_ids = count_image_ids(data)
#     print(f"Total 'image_id' occurrences (including duplicates): {total_image_ids}")

# except Exception as e:
#     print(f"An error occurred: {e}")

import json

# Load your JSON data
with open('attributes_sample6.json', 'r') as file:
    data = json.load(file)

# Extract image_id from each entry in the data
image_ids = [entry['image_id'] for entry in data]

# Get the total number of unique image_ids
unique_image_ids = len((image_ids))

# Print the total count of unique image_ids
print(f"Total number of unique image IDs: {unique_image_ids}")

