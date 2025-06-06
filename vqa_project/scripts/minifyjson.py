import json

def minify_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as f:
        json.dump(data, f, separators=(',', ':'), ensure_ascii=False)

# Example usage
input_file = "attributes_sample13.json"  # Replace with your input file
output_file = 'minified_attributes13.json'  # New file for minified JSON
minify_json(input_file, output_file)
