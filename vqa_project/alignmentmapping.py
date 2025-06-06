import json
import torch
import torch.nn as nn
from models.lxmert_vqa import LXMERTForVQA

# === Load original answer mappings ===
with open("D:/WORK/PG/Project/vqa_final/vqa_answer_mapping.json", "r") as f:
    ans_map_vqa = json.load(f)

with open("D:/WORK/PG/Project/vqa_final/vg_answer_mapping.json", "r") as f:
    ans_map_vg = json.load(f)

# === Load unified combined answer mapping ===
with open("combined_answer_map.json", "r") as f:
    answer_to_index_combined = json.load(f)

# === Load models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_combined_classes = len(answer_to_index_combined)

model_vqa = LXMERTForVQA(num_answers=num_combined_classes)
model_vqa.load_state_dict(torch.load("D:/WORK/PG/Project/vqa_final/vqa_model.pth", map_location="cpu"), strict=False)

model_vg = LXMERTForVQA(num_answers=num_combined_classes)
model_vg.load_state_dict(torch.load("D:/WORK/PG/Project/vqa_final/vg_model.pth", map_location="cpu"), strict=False)

# === Remapping function with debugging ===
def remap_classifier_weights(old_model, old_map, new_map, model_name=""):
    final_fc = old_model.classifier[-1]
    old_weight = final_fc.weight.data
    old_bias = final_fc.bias.data
    old_dim = old_weight.shape[1]

    new_num_classes = len(new_map)
    new_weight = torch.zeros((new_num_classes, old_dim))
    new_bias = torch.zeros(new_num_classes)

    print(f"Remapping classifier for {model_name}...")
    remap_count = 0

    for ans, old_idx in old_map.items():
        new_idx = new_map.get(ans, new_map["<unk>"])
        if old_idx < old_weight.size(0):
            new_weight[new_idx] = old_weight[old_idx]
            new_bias[new_idx] = old_bias[old_idx]
            remap_count += 1

    print(f"{model_name} — Remapped {remap_count} answers out of {len(old_map)}")

    # Replace classifier head
    old_model.classifier[-1] = nn.Linear(old_dim, new_num_classes)
    old_model.classifier[-1].weight.data = new_weight
    old_model.classifier[-1].bias.data = new_bias

    print(f"{model_name} classifier new shape: {old_model.classifier[-1].weight.shape}")
    return old_model

# === Remap and debug ===
model_vqa = remap_classifier_weights(model_vqa, ans_map_vqa, answer_to_index_combined, model_name="VQA")
model_vg = remap_classifier_weights(model_vg, ans_map_vg, answer_to_index_combined, model_name="VG")

# === Sanity check for a known answer ===
test_answer = "yes"
if test_answer in ans_map_vqa and test_answer in answer_to_index_combined:
    old_idx = ans_map_vqa[test_answer]
    new_idx = answer_to_index_combined[test_answer]
    print(f"Sanity check for '{test_answer}':")
    print("VQA weight sum (new index):", model_vqa.classifier[-1].weight.data[new_idx].sum().item())

# === Save models ===
torch.save(model_vqa.state_dict(), "D:/WORK/PG/Project/output/vqa_model_aligned.pth")
torch.save(model_vg.state_dict(), "D:/WORK/PG/Project/output/vg_model_aligned.pth")

print("\n✅ Both models have been realigned and saved.")
