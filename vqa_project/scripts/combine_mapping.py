import json

# # Load original answer mappings
# with open("D:/WORK/PG/Project/vqa_final/vqa_answer_mapping.json", "r") as f:
#     ans_map_vqa = json.load(f)

# with open("D:/WORK/PG/Project/vqa_final/vg_answer_mapping.json", "r") as f:
#     ans_map_vg = json.load(f)

# # Merge answer vocabularies
# all_answers = sorted(set(ans_map_vqa.keys()) | set(ans_map_vg.keys()))
# answer_to_index_combined = {ans: idx for idx, ans in enumerate(all_answers)}
# answer_to_index_combined["<unk>"] = len(answer_to_index_combined)

# # Save for future inference
# with open("combined_answer_map.json", "w") as f:
    # json.dump(answer_to_index_combined, f, indent=2)

# Fix combined_answer_map.json if needed

# with open("combined_answer_map.json", "r") as f:
#     ans_map = json.load(f)

# if "<unk>" not in ans_map:
#     ans_map["<unk>"] = len(ans_map)

# # Reindex all keys to ensure consistency
# sorted_answers = sorted([ans for ans in ans_map if ans != "<unk>"])
# answer_to_index = {ans: idx for idx, ans in enumerate(sorted_answers)}
# answer_to_index["<unk>"] = len(answer_to_index)

# # Save corrected mapping
# with open("combined_answer_map.json", "w") as f:
#     json.dump(answer_to_index, f, indent=2)

import json

# Paths to your existing answer mapping files
vqa_mapping_path = "D:/WORK/PG/Project/vqa_final/vqa_answer_mapping.json"
vg_mapping_path = "D:/WORK/PG/Project/vg_answer_mapping.json"

# Output path for combined mapping
combined_mapping_path = "D:/WORK/PG/Project/combined_answer_map.json"

# Load existing mappings
with open(vqa_mapping_path, "r") as f:
    vqa_map = json.load(f)

with open(vg_mapping_path, "r") as f:
    vg_map = json.load(f)

# Merge keys from both mappings
all_answers = set(vqa_map.keys()) | set(vg_map.keys())
all_answers.add("<unk>")  # Ensure unknown token exists

# Create new mapping
combined_map = {answer: idx for idx, answer in enumerate(sorted(all_answers))}

# Save combined mapping
with open(combined_mapping_path, "w") as f:
    json.dump(combined_map, f, indent=4)

print(f"✅ Combined mapping saved to: {combined_mapping_path}")
print(f"🔢 Total unique answers: {len(combined_map)}")

