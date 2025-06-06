import os
import json
import torch
import torch.nn as nn
from transformers import LxmertTokenizer
from models.lxmert_vqa import LXMERTForVQA

# === Paths ===
combined_answer_map_path = "D:/WORK/PG/Project/vqa_final/combined_answer_map.json"
vqa_model_path = "D:/WORK/PG/Project/vqa_final/vqa_model.pth"
vg_model_path = "D:/WORK/PG/Project/vqa_final/vgmodel.pth"
output_dir = "D:/WORK/PG/Project/output/fusion_models"

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

# Move to device
model_vqa.to(device)
model_vg.to(device)

print("Both models loaded successfully with shared answer space.")

# === Fusion Models ===

# 1. CBMS: Confidence-Based Model Switching
class CBMSModel(nn.Module):
    def __init__(self, model_vqa, model_vg):
        super().__init__()
        self.model_vqa = model_vqa
        self.model_vg = model_vg

    def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
        logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
        logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

        confidence_vqa = torch.max(torch.softmax(logits_vqa, dim=1), dim=1)[0]
        confidence_vg = torch.max(torch.softmax(logits_vg, dim=1), dim=1)[0]

        combined_logits = torch.where(
            confidence_vqa.unsqueeze(1) > confidence_vg.unsqueeze(1),
            logits_vqa,
            logits_vg
        )
        return combined_logits

# 2. Soft Voting: Averaging logits
class SoftVotingModel(nn.Module):
    def __init__(self, model_vqa, model_vg):
        super().__init__()
        self.model_vqa = model_vqa
        self.model_vg = model_vg

    def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
        logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
        logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

        combined_logits = (logits_vqa + logits_vg) / 2
        return combined_logits

# 3. Hard Voting: Majority vote on predictions
class HardVotingModel(nn.Module):
    def __init__(self, model_vqa, model_vg):
        super().__init__()
        self.model_vqa = model_vqa
        self.model_vg = model_vg

    def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
        logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
        logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

        preds_vqa = torch.argmax(logits_vqa, dim=1)
        preds_vg = torch.argmax(logits_vg, dim=1)

        # If predictions agree, keep that. Else default to VQA prediction (can customize)
        final_preds = torch.where(preds_vqa == preds_vg, preds_vqa, preds_vqa)

        # Convert to one-hot logits to fit return format
        combined_logits = torch.zeros_like(logits_vqa)
        combined_logits.scatter_(1, final_preds.unsqueeze(1), 1.0)
        return combined_logits

# === Save all fusion models ===
fusion_models = {
    "CBMS": (CBMSModel, "cbms_model.pth"),
    "SoftVoting": (SoftVotingModel, "soft_voting_model.pth"),
    "HardVoting": (HardVotingModel, "hard_voting_model.pth")
}

os.makedirs(output_dir, exist_ok=True)

for name, (cls, filename) in fusion_models.items():
    print(f"Building {name} model...")
    fusion_model = cls(model_vqa, model_vg).to(device)
    save_path = os.path.join(output_dir, filename)

    if not os.path.exists(save_path):
        torch.save(fusion_model.state_dict(), save_path)
        print(f"{name} model saved at: {save_path}")
    else:
        print(f"{name} model already exists at: {save_path} (not overwritten)")
