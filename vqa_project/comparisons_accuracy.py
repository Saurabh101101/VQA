import os
import json
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from transformers import LxmertTokenizer
from models.lxmert_vqa import LXMERTForVQA
from collections import Counter

# === Paths ===
combined_answer_map_path = "D:/WORK/PG/Project/vqa_final/combined_answer_map.json"
vqa_model_path = "D:/WORK/PG/Project/vqa_final/vqa_model.pth"
vg_model_path = "D:/WORK/PG/Project/vqa_final/vgmodel.pth"
sample_data_path = "D:/WORK/PG/Project/vqa_final/v2.0_sample.pkl"

# === Setup ===
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load answer mapping ===
with open(combined_answer_map_path, "r") as f:
    combined_answer_map = json.load(f)
index_to_answer = {v: k for k, v in combined_answer_map.items()}
num_answers = len(combined_answer_map)

# === Feature projector ===
class FeatureProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2048)

    def forward(self, x):
        return self.linear(x)

feature_projector = FeatureProjector().to(device)

# === Load models ===
model_vqa = LXMERTForVQA(num_answers)
model_vg = LXMERTForVQA(num_answers)
model_vqa.load_state_dict(torch.load(vqa_model_path, map_location="cpu")["model_state_dict"])
model_vg.load_state_dict(torch.load(vg_model_path, map_location="cpu")["model_state_dict"])
model_vqa.to(device).eval()
model_vg.to(device).eval()

# === Helper functions ===
def normalize_answer(ans):
    return ans.lower().strip()

def vqa_accuracy(pred, gt_answers):
    pred = normalize_answer(pred)
    gt_answers = [normalize_answer(a) for a in gt_answers]
    count = gt_answers.count(pred)
    return min(count / 3.0, 1.0)

# === Evaluation metrics ===
metrics = {
    "majority_vote": {"soft": 0, "hard": 0},
    "avg_logits": {"soft": 0, "hard": 0},
    "confidence": {"soft": 0, "hard": 0},
}
total = 0

# === Load dataset ===
with open(sample_data_path, "rb") as f:
    dataset = pickle.load(f)

# === Evaluation loop ===
for sample in tqdm(dataset, desc="Evaluating"):
    question = sample["question"]
    raw_feats = torch.tensor(sample["image_features"], dtype=torch.float).to(device)
    gt_answers = sample["answers"]
    most_common_answer = Counter([a.lower().strip() for a in gt_answers]).most_common(1)[0][0]

    encoding = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=20,
        return_tensors="pt"
    ).to(device)

    visual_feats = feature_projector(raw_feats).unsqueeze(0)
    visual_pos = torch.zeros_like(raw_feats).unsqueeze(0)

    with torch.no_grad():
        logits_vqa = model_vqa(
            input_ids=encoding["input_ids"],
            token_type_ids=encoding["token_type_ids"],
            attention_mask=encoding["attention_mask"],
            visual_feats=visual_feats,
            visual_pos=visual_pos
        )

        logits_vg = model_vg(
            input_ids=encoding["input_ids"],
            token_type_ids=encoding["token_type_ids"],
            attention_mask=encoding["attention_mask"],
            visual_feats=visual_feats,
            visual_pos=visual_pos
        )

        probs_vqa = torch.softmax(logits_vqa, dim=1)
        probs_vg = torch.softmax(logits_vg, dim=1)

        # === Confidence-based fusion ===
        conf_vqa = torch.max(probs_vqa, dim=1)[0].item()
        conf_vg = torch.max(probs_vg, dim=1)[0].item()
        conf_pred_idx = torch.argmax(logits_vqa if conf_vqa >= conf_vg else logits_vg, dim=1).item()
        conf_pred = index_to_answer.get(conf_pred_idx, "<unk>").strip().lower()

        # === Averaging logits ===
        avg_logits = (logits_vqa + logits_vg) / 2
        avg_pred_idx = torch.argmax(avg_logits, dim=1).item()
        avg_pred = index_to_answer.get(avg_pred_idx, "<unk>").strip().lower()

        # === Majority voting (based on individual predictions) ===
        vqa_pred_idx = torch.argmax(logits_vqa, dim=1).item()
        vg_pred_idx = torch.argmax(logits_vg, dim=1).item()
        vote_counter = Counter([vqa_pred_idx, vg_pred_idx])
        majority_pred_idx = vote_counter.most_common(1)[0][0]
        majority_pred = index_to_answer.get(majority_pred_idx, "<unk>").strip().lower()

    # === Update metrics ===
    for method, pred in zip(["majority_vote", "avg_logits", "confidence"], [conf_pred, avg_pred, majority_pred]):
        metrics[method]["soft"] += vqa_accuracy(pred, gt_answers)
        if pred == most_common_answer:
            metrics[method]["hard"] += 1

    total += 1

# === Final results ===
for method in metrics:
    soft = metrics[method]["soft"] * 100 / total
    hard = metrics[method]["hard"] * 100 / total
    print(f"\n🔹 {method.replace('_', ' ').title()} Fusion:")
    print(f"   Soft VQA Accuracy (Consensus): {soft:.2f}%")
    print(f"   Hard Accuracy (Exact Match):    {hard:.2f}%")
