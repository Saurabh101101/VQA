# # import torch
# # import pickle
# # from tqdm import tqdm
# # from models.lxmert_vqa import LXMERTForVQA
# # from transformers import LxmertTokenizer
# # from utils.data_loader import get_data_loaders
# # import torch.nn as nn

# # # Paths
# # val_file = "vqa_train.pkl"
# # checkpoint_path = "output/vqa_model/checkpoint.pth"

# # # Device setup
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # feature_projector = nn.Linear(4, 2048).to(device)

# # # Load validation dataset
# # _, val_loader = get_data_loaders(train_file=None, val_file=val_file, batch_size=8, num_workers=0)

# # # Load answer mapping
# # with open(val_file, 'rb') as f:
# #     val_dataset = pickle.load(f)
# # all_answers = [answer for sample in val_dataset for answer in sample['answers']]
# # unique_answers = list(set(all_answers))
# # answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}
# # num_answers = len(unique_answers)
# # answer_to_index['<unk>'] = len(answer_to_index)

# # # Load tokenizer
# # tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# # # Load model
# # model = LXMERTForVQA(num_answers=num_answers).to(device)
# # checkpoint = torch.load(checkpoint_path, map_location=device)
# # model.load_state_dict(checkpoint['model_state_dict'])
# # model.eval()

# # def resolve_answers(answers):
# #     if isinstance(answers[0], list):  # Nested list structure
# #         flat_answers = [item for sublist in answers for item in sublist]
# #         # print("Flattened Answers:", flat_answers)  # Debugging line
# #         if flat_answers:
# #             return max(set(flat_answers), key=flat_answers.count)  # Most frequent answer
# #         else:
# #             return None  # Handle empty lists gracefully
# #     else:  # Already flat
# #         # print("Flat Answers:", answers)  # Debugging line
# #         return max(set(answers), key=answers.count)

# # # Accuracy evaluation function
# # def evaluate_model():
# #     correct = 0
# #     total = 0

# #     with torch.no_grad():
# #         for batch in tqdm(val_loader, desc="Evaluating Model"):
# #             questions = batch['question']
# #             image_features = batch['image_features'].to(device)
# #             answers = batch['answers']

# #             # Tokenize questions
# #             encoding = tokenizer(
# #                 questions,
# #                 padding="max_length",
# #                 truncation=True,
# #                 max_length=20,
# #                 return_tensors="pt"
# #             ).to(device)

# #             visual_feats = feature_projector(image_features.view(-1, 4)).view(
# #                 image_features.shape[0], image_features.shape[1], 2048
# #             )
# #             visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

# #             # Forward pass
# #             outputs = model(
# #                 input_ids=encoding['input_ids'],
# #                 token_type_ids=encoding['token_type_ids'],
# #                 attention_mask=encoding['attention_mask'],
# #                 visual_feats=visual_feats,
# #                 visual_pos=visual_pos
# #             )

# #             # Apply softmax to get probabilities
# #             probs = torch.softmax(outputs, dim=1)

# #             # Get predictions
# #             _, predicted = torch.max(probs, dim=1)

# #             # Resolve actual answers
# #             resolved_answers = [
# #                 answer_to_index.get(resolve_answers(a), answer_to_index['<unk>'])
# #                 for a in answers
# #             ]

# #             labels = torch.tensor(resolved_answers, dtype=torch.long).to(device)

# #             total += labels.size(0)
# #             correct += (predicted == labels).sum().item()

# #             # Print predicted and actual answers for comparison
# #             print(f"Predicted indices: {predicted.cpu().numpy()}")
# #             print(f"Resolved indices: {resolved_answers}")
# #             print(f"Predicted answers: {[unique_answers[idx] for idx in predicted.cpu().numpy()]}")
# #             print(f"Actual answers: {[unique_answers[idx] for idx in resolved_answers]}")

# #     accuracy = correct / total if total > 0 else 0
# #     print(f"Model Accuracy: {accuracy:.4f}")




# # # Run evaluation
# # evaluate_model()


# import torch
# import pickle
# import json
# import os
# from tqdm import tqdm
# from models.lxmert_vqa import LXMERTForVQA
# from transformers import LxmertTokenizer
# import torch.nn as nn

# # Paths
# pkl_file = "D:/WORK/PG/Project/vqa_final/v2.0_sample.pkl"  # Change to your dataset file if needed
# answer_mapping_file = "vg_answer_mapping.json"
# checkpoint_path = "D:/WORK/PG/Project/output/vqa_model/checkpoint.pth"

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feature_projector = nn.Linear(4, 2048).to(device)
# # Load the answer mapping
# with open(answer_mapping_file, "r") as f:
#     answer_to_index = json.load(f)
# index_to_answer = {v: k for k, v in answer_to_index.items()}  # Reverse mapping

# class CombineLXMERT(nn.Module):
#     def __init__(self, model_vqa, model_vg):
#         super(CombineLXMERT, self).__init__()
#         self.model_vqa = model_vqa
#         self.model_vg = model_vg

#     def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
#         logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
#         logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

#         # Compute confidence scores
#         confidence_vqa = torch.max(torch.softmax(logits_vqa, dim=1), dim=1)[0]
#         confidence_vg = torch.max(torch.softmax(logits_vg, dim=1), dim=1)[0]

#         # Select the model with higher confidence
#         combined_logits = torch.where(confidence_vqa.unsqueeze(1) > confidence_vg.unsqueeze(1), logits_vqa, logits_vg)

#         return combined_logits

# # Load model
# num_answers = len(answer_to_index)
# model = LXMERTForVQA(num_answers=num_answers).to(device)
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# # Load tokenizer
# tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# # Load dataset
# with open(pkl_file, "rb") as f:
#     dataset = pickle.load(f)

# # Define function to get most frequent answer
# def resolve_answers(answers):
#     if isinstance(answers[0], list):
#         flat_answers = [item for sublist in answers for item in sublist]
#         return max(set(flat_answers), key=flat_answers.count) if flat_answers else "<unk>"
#     else:
#         return max(set(answers), key=answers.count)

# # Evaluation
# correct = 0
# total = 0

# with torch.no_grad():
#     for sample in tqdm(dataset, desc="Evaluating"):
#         question = sample["question"]
#         image_features = torch.tensor(sample["image_features"]).unsqueeze(0).to(device)
#         true_answer = resolve_answers(sample["answers"])

#         # Tokenize the question
#         encoding = tokenizer(
#             question, padding="max_length", truncation=True, max_length=20, return_tensors="pt"
#         ).to(device)

#         visual_feats = feature_projector(image_features.view(-1, 4)).view(
#             image_features.shape[0], image_features.shape[1], 2048
#         )
#         visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

#         # Forward pass
#         outputs = model(
#             input_ids=encoding["input_ids"],
#             token_type_ids=encoding["token_type_ids"],
#             attention_mask=encoding["attention_mask"],
#             visual_feats=visual_feats,
#             visual_pos=visual_pos
#         )

#         # Get predicted answer
#         predicted_index = outputs.argmax(dim=1).item()
#         predicted_answer = index_to_answer.get(predicted_index, "<unk>")

#         # Compare with ground truth
#         if predicted_answer == true_answer:
#             correct += 1
#         total += 1

# # Compute accuracy
# accuracy = correct / total if total > 0 else 0
# print(f"Model Accuracy: {accuracy:.4f}")




# ######################################################################################################################################################################################################################
# ######################################################################################################################################################################################################################
# ######################################################################################################################################################################################################################
# ######################################################################################################################################################################################################################



# # import torch
# # import torch.nn as nn
# # import pickle
# # import gc
# # from tqdm import tqdm
# # from transformers import LxmertTokenizer
# # from models.lxmert_vqa import LXMERTForVQA
# # from utils.data_loader import get_data_loaders

# # # Load tokenizer
# # tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# # # Detect device
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # feature_projector = nn.Linear(4, 2048).to(device)

# # # Paths to training data
# # train_vqa_file = "vqa_train.pkl"
# # train_vg_file = "vg_train.pkl"

# # def get_answer_mapping(train_file):
# #     """Load answer mappings from training data to determine the number of output classes."""
# #     with open(train_file, "rb") as f:
# #         train_dataset = pickle.load(f)

# #     all_answers = [answer for sample in train_dataset for answer in sample["answers"]]
# #     unique_answers = sorted(set(all_answers))  # Ensure order consistency
# #     answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}

# #     # Add unknown answer mapping
# #     answer_to_index["<unk>"] = len(answer_to_index)

# #     return len(unique_answers), answer_to_index

# # # Compute number of answers dynamically
# # num_answers_vqa, answer_to_index_vqa = get_answer_mapping(train_vqa_file)
# # num_answers_vg, answer_to_index_vg = get_answer_mapping(train_vg_file)

# # print(f"VQA Classes: {num_answers_vqa}, VG Classes: {num_answers_vg}")

# # # Paths to validation dataset
# # val_vqa_file = "vqa_val.pkl"
# # val_vg_file = "vg_val.pkl"

# # # Load validation datasets
# # _, val_vqa_loader = get_data_loaders(train_file=None, val_file=val_vqa_file, batch_size=8, num_workers=0)
# # _, val_vg_loader = get_data_loaders(train_file=None, val_file=val_vg_file, batch_size=8, num_workers=0)

# # # Define the Combined Model (Must match the saved structure)
# # class CombinedLXMERT(nn.Module):
# #     def __init__(self, model_vqa, model_vg):
# #         super(CombinedLXMERT, self).__init__()
# #         self.model_vqa = model_vqa
# #         self.model_vg = model_vg

# #     def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
# #         logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
# #         logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

# #         # Compute confidence scores
# #         confidence_vqa = torch.max(torch.softmax(logits_vqa, dim=1), dim=1)[0]
# #         confidence_vg = torch.max(torch.softmax(logits_vg, dim=1), dim=1)[0]

# #         # Select the model with higher confidence
# #         combined_logits = torch.where(confidence_vqa.unsqueeze(1) > confidence_vg.unsqueeze(1), logits_vqa, logits_vg)

# #         return combined_logits

# # # Load base models
# # model_vqa = LXMERTForVQA(num_answers=num_answers_vqa).to("cpu")
# # model_vg = LXMERTForVQA(num_answers=num_answers_vg).to("cpu")

# # # Load combined model
# # combined_model = CombinedLXMERT(model_vqa, model_vg)
# # combined_model.load_state_dict(torch.load("D:/WORK/PG/Project/output/combined_model.pth", map_location="cpu"))
# # combined_model.to(device)
# # combined_model.eval()

# # print("Combined model loaded successfully.")

# # def resolve_answers(answers):
# #     """Handles both nested and flat answer lists by selecting the most frequent answer."""
# #     if isinstance(answers[0], list):  # If answers are in nested lists
# #         flat_answers = [item for sublist in answers for item in sublist]
# #         return max(set(flat_answers), key=flat_answers.count) if flat_answers else "<unk>"
# #     return max(set(answers), key=answers.count)

# # # Define inference function
# # def run_inference(model, dataloader, answer_to_index, task_name):
# #     model.eval()
# #     correct = 0
# #     total = 0
# #     running_loss = 0.0
# #     criterion = nn.CrossEntropyLoss()

# #     with torch.no_grad():
# #         for batch in tqdm(dataloader, desc=f"Evaluating {task_name.upper()}"):
# #             questions = batch["question"]
# #             image_features = batch["image_features"].to(device)
# #             answers = batch["answers"]

# #             # Tokenize questions
# #             encoding = tokenizer(
# #                 questions,
# #                 padding="max_length",
# #                 truncation=True,
# #                 max_length=20,
# #                 return_tensors="pt"
# #             ).to(device)

# #             # Process visual features
# #             visual_feats = feature_projector(image_features.view(-1, 4)).view(
# #                 image_features.shape[0], image_features.shape[1], 2048
# #             )
# #             visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

# #             # Convert answers to indices
# #             resolved_answers = [
# #                 answer_to_index.get(resolve_answers(a), answer_to_index["<unk>"])
# #                 for a in answers
# #             ]
# #             labels = torch.tensor(resolved_answers, dtype=torch.long).to(device)

# #             # **Filter out invalid labels**
# #             valid_indices = (labels >= 0) & (labels < len(answer_to_index))
# #             if valid_indices.sum() == 0:
# #                 print("Skipping batch due to invalid labels.")
# #                 continue

# #             labels = labels[valid_indices]
# #             encoding = {k: v[valid_indices] for k, v in encoding.items()}
# #             visual_feats = visual_feats[valid_indices]
# #             visual_pos = visual_pos[valid_indices]

# #             # Forward pass
# #             outputs = model(
# #                 input_ids=encoding["input_ids"],
# #                 token_type_ids=encoding["token_type_ids"],
# #                 attention_mask=encoding["attention_mask"],
# #                 visual_feats=visual_feats,
# #                 visual_pos=visual_pos
# #             )

# #             loss = criterion(outputs, labels)
# #             running_loss += loss.item()

# #             # Compute accuracy
# #             _, predicted = torch.max(outputs, dim=1)
# #             total += labels.size(0)
# #             correct += (predicted == labels).sum().item()

# #             del questions, image_features, encoding, visual_feats, visual_pos, labels, outputs
# #             torch.cuda.empty_cache()
# #             gc.collect()

# #     accuracy = correct / total if total > 0 else 0
# #     avg_loss = running_loss / len(dataloader)
# #     print(f"{task_name.upper()} Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
# #     return accuracy

# # # Run inference on VQA and VG datasets
# # accuracy_vqa = run_inference(combined_model, val_vqa_loader, answer_to_index_vqa, "vqa")
# # accuracy_vg = run_inference(combined_model, val_vg_loader, answer_to_index_vg, "vg")

# # print(f"Final Combined Model Accuracy - VQA: {accuracy_vqa:.4f}, VG: {accuracy_vg:.4f}")

# ##############################################################################################################################

# # import torch
# # import pickle
# # import json
# # import os
# # from tqdm import tqdm
# # from models.lxmert_vqa import LXMERTForVQA
# # from transformers import LxmertTokenizer
# # import torch.nn as nn

# # # Paths
# # pkl_file = "sample5_1.pkl"  # Change to your dataset file if needed
# # answer_mapping_file = "ogvqa_answer_mapping.json"
# # checkpoint_path = "output/combined_model.pth"  # Use combined model

# # # Device setup
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # feature_projector = nn.Linear(4, 2048).to(device)

# # # Load answer mappings
# # with open(answer_mapping_file, "r") as f:
# #     answer_to_index = json.load(f)
# # index_to_answer = {v: k for k, v in answer_to_index.items()}  # Reverse mapping

# # # Load model parameters
# # num_answers = len(answer_to_index)

# # # Define Combined Model
# # class CombinedLXMERT(nn.Module):
# #     def __init__(self, model_vqa, model_vg):
# #         super(CombinedLXMERT, self).__init__()
# #         self.model_vqa = model_vqa
# #         self.model_vg = model_vg

# #     def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
# #         logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
# #         logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

# #         # Compute confidence scores
# #         confidence_vqa = torch.max(torch.softmax(logits_vqa, dim=1), dim=1)[0]
# #         confidence_vg = torch.max(torch.softmax(logits_vg, dim=1), dim=1)[0]

# #         # Select the model with higher confidence
# #         combined_logits = torch.where(confidence_vqa.unsqueeze(1) > confidence_vg.unsqueeze(1), logits_vqa, logits_vg)

# #         return combined_logits

# # # Load base models
# # model_vqa = LXMERTForVQA(num_answers=num_answers).to("cpu")
# # model_vg = LXMERTForVQA(num_answers=num_answers).to("cpu")

# # # Load combined model
# # combined_model = CombinedLXMERT(model_vqa, model_vg)
# # checkpoint = torch.load(checkpoint_path, map_location="cpu")
# # combined_model.load_state_dict(checkpoint)
# # combined_model.to(device)
# # combined_model.eval()

# # print("Combined model loaded successfully.")

# # # Load tokenizer
# # tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# # # Load dataset
# # with open(pkl_file, "rb") as f:
# #     dataset = pickle.load(f)

# # # Define function to get most frequent answer
# # def resolve_answers(answers):
# #     if isinstance(answers[0], list):
# #         flat_answers = [item for sublist in answers for item in sublist]
# #         return max(set(flat_answers), key=flat_answers.count) if flat_answers else "<unk>"
# #     else:
# #         return max(set(answers), key=answers.count)

# # # Evaluation
# # correct = 0
# # total = 0

# # with torch.no_grad():
# #     for sample in tqdm(dataset, desc="Evaluating"):
# #         question = sample["question"]
# #         image_features = torch.tensor(sample["image_features"]).unsqueeze(0).to(device)
# #         true_answer = resolve_answers(sample["answers"])

# #         # Tokenize the question
# #         encoding = tokenizer(
# #             question, padding="max_length", truncation=True, max_length=20, return_tensors="pt"
# #         ).to(device)

# #         visual_feats = feature_projector(image_features.view(-1, 4)).view(
# #             image_features.shape[0], image_features.shape[1], 2048
# #         )
# #         visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

# #         # Forward pass
# #         outputs = combined_model(
# #             input_ids=encoding["input_ids"],
# #             token_type_ids=encoding["token_type_ids"],
# #             attention_mask=encoding["attention_mask"],
# #             visual_feats=visual_feats,
# #             visual_pos=visual_pos
# #         )

# #         # Get predicted answer
# #         predicted_index = outputs.argmax(dim=1).item()
# #         predicted_answer = index_to_answer.get(predicted_index, "<unk>")

# #         # Compare with ground truth
# #         if predicted_answer == true_answer:
# #             correct += 1
# #         total += 1

# # # Compute accuracy
# # accuracy = correct / total if total > 0 else 0
# # print(f"Combined Model Accuracy: {accuracy:.4f}")


# import torch
# import torch.nn as nn
# import pickle
# from torch.utils.data import Dataset, DataLoader
# from transformers import LxmertTokenizer
# from models.lxmert_vqa import LXMERTForVQA
# from models.lxmert_vg import LXMERTWithSceneGraph
# import os
# from tqdm import tqdm

# # Setup device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Feature projector (from 4D to 2048D)
# feature_projector = nn.Linear(4, 2048).to(device)

# # Load tokenizer
# tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# # Load combined answer mapping
# with open("combined_answer_map.json", "r") as f:
#     import json
#     combined_answer_map = json.load(f)
# index_to_answer = {idx: ans for ans, idx in combined_answer_map.items()}

# # === Custom Dataset ===
# class VQADataset(Dataset):
#     def __init__(self, pkl_file, tokenizer):
#         with open(pkl_file, "rb") as f:
#             self.data = pickle.load(f)
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         question = sample["question"]
#         answers = sample["answers"]  # List of human answers
#         visual_feats = torch.tensor(sample["image_features"], dtype=torch.float)  # Shape: (N, 4)

#         # Tokenize question
#         encoding = self.tokenizer(
#             question,
#             padding="max_length",
#             truncation=True,
#             max_length=20,
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "token_type_ids": encoding["token_type_ids"].squeeze(0),
#             "visual_feats": visual_feats,
#             "visual_pos": visual_feats.clone(),  # Use same for position (optional)
#             "answers": answers
#         }

# # === Load Models ===
# num_classes = len(combined_answer_map)
# model_vqa = LXMERTForVQA(num_classes)
# model_vg = LXMERTForVQA(num_classes)

# model_vqa.load_state_dict(torch.load("D:/WORK/PG/Project/output/vqa_model_aligned.pth", map_location="cpu"))
# model_vg.load_state_dict(torch.load("D:/WORK/PG/Project/output/vg_model_aligned.pth", map_location="cpu"))

# model_vqa.to(device)
# model_vg.to(device)

# # === Combined Model ===
# class CombinedLXMERT(nn.Module):
#     def __init__(self, model_vqa, model_vg):
#         super().__init__()
#         self.model_vqa = model_vqa
#         self.model_vg = model_vg

#     def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
#         logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
#         logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

#         conf_vqa = torch.max(torch.softmax(logits_vqa, dim=1), dim=1)[0]
#         conf_vg = torch.max(torch.softmax(logits_vg, dim=1), dim=1)[0]

#         combined_logits = torch.where(
#             conf_vqa.unsqueeze(1) > conf_vg.unsqueeze(1), logits_vqa, logits_vg
#         )

#         return combined_logits

# combined_model = CombinedLXMERT(model_vqa, model_vg).to(device)
# combined_model.eval()

# # === Evaluation Loop ===
# dataset = VQADataset("D:/WORK/PG/Project/vqa_final/v2.0_sample.pkl", tokenizer)
# data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# correct = 0
# total = 0

# with torch.no_grad():
#     for batch in tqdm(data_loader):
#         input_ids = batch["input_ids"].to(device)          # shape: (1, seq_len)
#         attention_mask = batch["attention_mask"].to(device)
#         token_type_ids = batch["token_type_ids"].to(device)

#         feats = batch["visual_feats"].to(device)           # shape: (1, N, 4)
#         visual_feats = feature_projector(feats)            # (1, N, 2048)
#         visual_pos = feats                                 # (1, N, 4)

#         logits = combined_model(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
#         predicted = torch.argmax(logits, dim=1).item()
#         predicted_answer = index_to_answer[predicted]

#         gt_answers = [ans.lower() for ans in batch["answers"][0]]
#         if predicted_answer in gt_answers:
#             correct += 1
#         total += 1


# accuracy = correct / total * 100
# print(f"Accuracy on evaluation set: {accuracy:.2f}%")

import os
import json
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from transformers import LxmertTokenizer
from models.lxmert_vqa import LXMERTForVQA
from collections import Counter

# Paths
combined_answer_map_path = "D:/WORK/PG/Project/vqa_final/combined_answer_map.json"
vqa_model_path = "D:/WORK/PG/Project/output/vqa_model/checkpoint.pth"
vg_model_path = "D:/WORK/PG/Project/output/vgmodel.pth"
combined_model_path = "D:/WORK/PG/Project/output/Fmodel.pth"
sample_data_path = "D:/WORK/PG/Project/vqa_final/vg_sample.pkl"

# Setup
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load answer mapping
with open(combined_answer_map_path, "r") as f:
    combined_answer_map = json.load(f)
index_to_answer = {v: k for k, v in combined_answer_map.items()}
num_answers = len(combined_answer_map)

# Feature projector
class FeatureProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2048)

    def forward(self, x):
        return self.linear(x)

feature_projector = FeatureProjector().to(device)

# Combined Model
class CombinedLXMERT(nn.Module):
    def __init__(self, model_vqa, model_vg):
        super().__init__()
        self.model_vqa = model_vqa
        self.model_vg = model_vg

    def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
        logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
        logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

        conf_vqa = torch.max(torch.softmax(logits_vqa, dim=1), dim=1)[0]
        conf_vg = torch.max(torch.softmax(logits_vg, dim=1), dim=1)[0]

        return {
            "logits_vqa": logits_vqa,
            "logits_vg": logits_vg,
            "conf_vqa": conf_vqa,
            "conf_vg": conf_vg
        }

# Load models
model_vqa = LXMERTForVQA(num_answers)
model_vg = LXMERTForVQA(num_answers)
model_vqa.load_state_dict(torch.load(vqa_model_path, map_location="cpu")["model_state_dict"])
model_vg.load_state_dict(torch.load(vg_model_path, map_location="cpu")["model_state_dict"])
model_vqa.to(device)
model_vg.to(device)

combined_model = CombinedLXMERT(model_vqa, model_vg).to(device)
combined_model.load_state_dict(torch.load(combined_model_path, map_location=device))
combined_model.eval()

# Load data
with open(sample_data_path, "rb") as f:
    dataset = pickle.load(f)

# Helper functions
def normalize_answer(ans):
    return ans.lower().strip()

def vqa_accuracy(pred, gt_answers):
    """Compute VQA soft accuracy as per VQA v2.0 standard."""
    pred = normalize_answer(pred)
    gt_answers = [normalize_answer(a) for a in gt_answers]
    count = gt_answers.count(pred)
    return min(count / 3.0, 1.0)

# Evaluation loop
soft_correct = 0
hard_correct = 0
total = 0

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
        outputs = combined_model(
            input_ids=encoding["input_ids"],
            token_type_ids=encoding["token_type_ids"],
            attention_mask=encoding["attention_mask"],
            visual_feats=visual_feats,
            visual_pos=visual_pos
        )

        if outputs["conf_vqa"].item() >= outputs["conf_vg"].item():
            pred_idx = torch.argmax(outputs["logits_vqa"], dim=1).item()
        else:
            pred_idx = torch.argmax(outputs["logits_vg"], dim=1).item()

        pred_answer = index_to_answer.get(pred_idx, "<unk>").strip().lower()

    # Soft and Hard accuracy
    soft_acc = vqa_accuracy(pred_answer, gt_answers)
    soft_correct += soft_acc
    if pred_answer == most_common_answer:
        hard_correct += 1
    total += 1

# Final metrics
print(f"Soft VQA Accuracy (Consensus): {100 * soft_correct / total:.2f}%")
print(f"Hard Accuracy (Exact Match):    {100 * hard_correct / total:.2f}%")




