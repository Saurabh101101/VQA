# import json
# import pickle

# def json_to_pkl(json_file_path, pkl_file_path):
#     try:
#         # Load JSON data
#         with open(json_file_path, 'r') as json_file:
#             data = json.load(json_file)
        
#         # Save data as PKL
#         with open(pkl_file_path, 'wb') as pkl_file:
#             pickle.dump(data, pkl_file)
        
#         print(f"Successfully converted {json_file_path} to {pkl_file_path}")
#     except Exception as e:
#         print(f"Error: {e}")

# # Example usage
# json_file_path = "D:/WORK/PG/Project/vqa_project/data/processed/minified_relationships.json"  # Replace with your JSON file path
# pkl_file_path = "currrelationships.pkl"    # Replace with your desired PKL file path
# json_to_pkl(json_file_path, pkl_file_path)


# # import pickle

# # with open("D:/WORK/PG/Project/attributes.pkl" , "rb") as f:
# #     data = pickle.load(f)
# #     print(type(data))
# #     if isinstance(data, list):
# #         print(f"First element type: {(data[0])}")


import pickle
import json

def convert_pkl_to_minified_json(pkl_file_path, json_file_path):
    try:
        # Load the .pkl file
        with open(pkl_file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        # Convert and save as minified JSON
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, separators=(",", ":"))

        print(f"Successfully converted {pkl_file_path} to minified JSON at {json_file_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Example usage
# Replace 'example.pkl' and 'output.json' with your actual file paths
pkl_file = "D:/WORK/PG/Project/vqa_project/data/processed/currrelationships.pkl"
json_file = 'output.json'
convert_pkl_to_minified_json(pkl_file, json_file)
