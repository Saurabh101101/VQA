import os
import json
import torch
import torch.nn as nn
from transformers import LxmertTokenizer
from models.lxmert_vqa import LXMERTForVQA

# === Paths ===
combined_answer_map_path = "D:/WORK/PG/Project/combined_answer_map.json"
vqa_model_path = "D:/WORK/PG/Project/output/vqa_model/checkpoint.pth"
vg_model_path = "D:/WORK/PG/Project/output/vgmodel.pth"
output_model_path = "D:/WORK/PG/Project/output/Fmodel.pth"

# === Load tokenizer and device ===
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load combined answer map ===
with open(combined_answer_map_path, "r") as f:
    combined_answer_map = json.load(f)
num_classes = len(combined_answer_map)

# === Load both models ===
model_vqa = LXMERTForVQA(num_classes).to("cpu")
model_vg = LXMERTForVQA(num_classes).to("cpu")

vqa_state = torch.load(vqa_model_path, map_location="cpu")
vg_state = torch.load(vg_model_path, map_location="cpu")

model_vqa.load_state_dict(vqa_state["model_state_dict"])
model_vg.load_state_dict(vg_state["model_state_dict"])

# === Move to device ===
model_vqa.to(device)
model_vg.to(device)

print("✅ Both models loaded successfully with shared answer space.")

# === Define Combined Model ===
class CombinedLXMERT(nn.Module):
    def __init__(self, model_vqa, model_vg):
        super().__init__()
        self.model_vqa = model_vqa
        self.model_vg = model_vg

    def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
        logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
        logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

        confidence_vqa = torch.max(torch.softmax(logits_vqa, dim=1), dim=1)[0]
        confidence_vg = torch.max(torch.softmax(logits_vg, dim=1), dim=1)[0]

        # Combine based on confidence
        combined_logits = torch.where(
            confidence_vqa.unsqueeze(1) > confidence_vg.unsqueeze(1),
            logits_vqa,
            logits_vg
        )
        return combined_logits

# === Build and save combined model ===
combined_model = CombinedLXMERT(model_vqa, model_vg).to(device)

if not os.path.exists(output_model_path):
    torch.save(combined_model.state_dict(), output_model_path)
    print(f"✅ Combined model saved to: {output_model_path}")
else:
    print(f"ℹ️ Combined model already exists at: {output_model_path} (not overwritten)")

