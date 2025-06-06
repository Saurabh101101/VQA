# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from models.lxmert_vqa import LXMERTForVQA
# from transformers import LxmertTokenizer
# from torch.amp import GradScaler, autocast
# import gc
# import pickle

# from utils.data_loader import get_data_loaders

# # Paths to preprocessed files
# train_file = "vqa_train.pkl"
# val_file = "vqa_val.pkl"

# # Configuration
# batch_size = 8
# num_epochs = 10
# learning_rate = 1e-4
# output_dir = "output/vqa_model"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# accumulation_steps = 4
# checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Initialize DataLoaders
# train_loader, val_loader = get_data_loaders(train_file, val_file, batch_size=batch_size, num_workers=0)

# # Generate Answer Mapping
# with open(train_file, 'rb') as f:
#     train_dataset = pickle.load(f)
# all_answers = [answer for sample in train_dataset for answer in sample['answers']]
# unique_answers = list(set(all_answers))
# answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}
# num_answers = len(unique_answers)
# answer_to_index['<unk>'] = len(answer_to_index)

# tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
# # Model, Loss, Optimizer
# model = LXMERTForVQA(num_answers=num_answers).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# feature_projector = nn.Linear(4, 2048).to(device)
# scaler = GradScaler()

# # Load checkpoint if exists
# start_epoch = 1
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     print(f"Resumed training from epoch {start_epoch}")
#     optimizer.zero_grad()
#     torch.cuda.empty_cache()
#     gc.collect()

# def resolve_answers(answers):
#     if isinstance(answers[0], list):  # Nested list structure
#         flat_answers = [item for sublist in answers for item in sublist]
#         if flat_answers:
#             return max(set(flat_answers), key=flat_answers.count)  # Most frequent answer
#         else:
#             return None  # Handle empty lists gracefully
#     else:  # Already flat
#         return max(set(answers), key=answers.count)

# # Training Function
# def train_one_epoch(epoch):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     optimizer.zero_grad()

#     for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
#         questions = batch['question']
#         image_features = batch['image_features'].to(device)
#         answers = batch['answers']

#         # Tokenize questions
#         encoding = tokenizer(
#             questions,
#             padding="max_length",
#             truncation=True,
#             max_length=20,
#             return_tensors="pt"
#         ).to(device)

#         visual_feats = feature_projector(image_features.view(-1, 4)).view(
#             image_features.shape[0], image_features.shape[1], 2048)
#         visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

#         labels = torch.tensor(
#             [answer_to_index.get(resolve_answers(a), -1) for a in answers],
#             dtype=torch.long
#         ).to(device)

#         with autocast(device_type=device.type):
#             outputs = model(
#                 input_ids=encoding['input_ids'],
#                 token_type_ids=encoding['token_type_ids'],
#                 attention_mask=encoding['attention_mask'],
#                 visual_feats=visual_feats,
#                 visual_pos=visual_pos
#             )
#             loss = criterion(outputs, labels) / accumulation_steps

#         scaler.scale(loss).backward()
#         if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()

#         running_loss += loss.item() * accumulation_steps
#         _, predicted = torch.max(outputs, dim=1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         del questions, image_features, encoding, visual_feats, visual_pos, labels, outputs
#         torch.cuda.empty_cache()
#         gc.collect()

#     print(f"Epoch {epoch}, Loss: {running_loss:.4f}, Accuracy: {correct / total:.4f}")


# def validate():
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc="Validating"):
#             try:
#                 # Extract batch data
#                 questions = batch['question']
#                 image_features = batch['image_features'].to(device)
#                 answers = batch['answers']

#                 # Tokenize questions
#                 encoding = tokenizer(
#                     questions,
#                     padding="max_length",
#                     truncation=True,
#                     max_length=20,
#                     return_tensors="pt"
#                 ).to(device)

#                 # Project image features
#                 visual_feats = feature_projector(image_features.view(-1, 4)).view(
#                     image_features.shape[0], image_features.shape[1], 2048
#                 )
#                 visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

#                 # Resolve answers to indices
#                 resolved_answers = [
#                     answer_to_index.get(resolve_answers(a), answer_to_index['<unk>'])
#                     for a in answers
#                 ]
#                 labels = torch.tensor(resolved_answers, dtype=torch.long).to(device)

#                 # Filter valid labels
#                 valid_indices = (labels >= 0) & (labels < num_answers)
#                 if valid_indices.sum() == 0:
#                     print("Skipping batch due to invalid labels.")
#                     continue

#                 # Apply filtering to tensors
#                 labels = labels[valid_indices]
#                 encoding = {k: v[valid_indices] for k, v in encoding.items()}
#                 image_features = image_features[valid_indices]
#                 visual_feats = visual_feats[valid_indices]
#                 visual_pos = visual_pos[valid_indices]

#                 # Forward pass
#                 with autocast(device_type=device.type):
#                     outputs = model(
#                         input_ids=encoding['input_ids'],
#                         token_type_ids=encoding['token_type_ids'],
#                         attention_mask=encoding['attention_mask'],
#                         visual_feats=visual_feats,
#                         visual_pos=visual_pos
#                     )
#                     loss = criterion(outputs, labels)

#                 running_loss += loss.item()
#                 _, predicted = torch.max(outputs, dim=1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#             except Exception as e:
#                 print(f"Error in batch: {e}")
#                 continue

#             # Free memory
#             del questions, image_features, labels, outputs
#             torch.cuda.empty_cache()
#             gc.collect()

#     print(f"Validation Loss: {running_loss:.4f}, Accuracy: {correct / total:.4f}")

    
# # Main Training Loop
# if __name__ == "__main__":
#     for epoch in range(start_epoch, num_epochs + 1):
#         train_one_epoch(epoch)
#         validate()

#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#         }, checkpoint_path)
#         print(f"Checkpoint saved for epoch {epoch}")
#     print("Training complete!")


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.lxmert_vqa import LXMERTForVQA
from transformers import LxmertTokenizer
from torch.amp import GradScaler, autocast
import gc
import pickle
import json

from utils.data_loader import get_data_loaders

# Paths to preprocessed files and answer mapping file
train_file = "D:/WORK/PG/Project/vqa_final/vqa_train.pkl"
val_file = "D:/WORK/PG/Project/vqa_final/vqa_val.pkl"
answer_mapping_file = "combined_answer_map.json"

# Configuration
batch_size = 8
num_epochs = 10
learning_rate = 1e-4
output_dir = "output/vqa_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accumulation_steps = 4
checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize DataLoaders
train_loader, val_loader = get_data_loaders(train_file, val_file, batch_size=batch_size, num_workers=0)

# Load or generate Answer Mapping
if os.path.exists(answer_mapping_file):
    with open(answer_mapping_file, 'r') as f:
        answer_to_index = json.load(f)
    num_answers = len(answer_to_index)
    print(f"Loaded answer mapping from {answer_mapping_file}")
else:
    if os.path.exists(train_file):
        with open(train_file, 'rb') as f:
            train_dataset = pickle.load(f)
        
        # Extract all answers
        all_answers = []
        for sample in train_dataset:
            if 'answers' in sample:
                if isinstance(sample['answers'], list):
                    all_answers.extend(sample['answers'])
                else:
                    all_answers.append(sample['answers'])
        
        # Get unique answers
        unique_answers = list(set(all_answers))
        
        # Create answer-to-index mapping
        answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}
        answer_to_index['<unk>'] = len(answer_to_index)  # For unknown answers
        num_answers = len(answer_to_index)
        
        # Save mapping to JSON file
        with open(answer_mapping_file, 'w') as f:
            json.dump(answer_to_index, f, indent=4)
        print(f"Generated and saved answer mapping to {answer_mapping_file}")
    else:
        print(f"Error: {train_file} not found. Cannot generate answer mapping.")


tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# Model, Loss, Optimizer
model = LXMERTForVQA(num_answers=num_answers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
feature_projector = nn.Linear(4, 2048).to(device)
scaler = GradScaler()

# Load checkpoint if exists
start_epoch = 1
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed training from epoch {start_epoch}")
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()

def resolve_answers(answers):
    if not answers:  # Handle empty answers
        return '<unk>'
    if isinstance(answers[0], list):  # Nested list structure
        flat_answers = [item for sublist in answers for item in sublist if item is not None]
        if flat_answers:
            return max(set(flat_answers), key=flat_answers.count)  # Most frequent answer
        return '<unk>'
    else:  # Already flat
        valid_answers = [a for a in answers if a is not None]
        if valid_answers:
            return max(set(valid_answers), key=valid_answers.count)
        return '<unk>'

# Training Function
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        questions = batch['question']
        image_features = batch['image_features'].to(device)
        answers = batch['answers']

        # Tokenize questions
        encoding = tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=20,
            return_tensors="pt"
        ).to(device)

        visual_feats = feature_projector(image_features.view(-1, 4)).view(
            image_features.shape[0], image_features.shape[1], 2048)
        visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

        # Compute labels with <unk> as fallback
        labels = torch.tensor(
            [answer_to_index.get(resolve_answers(a), answer_to_index['<unk>']) for a in answers],
            dtype=torch.long
        ).to(device)

        # Verify label range
        if (labels < 0).any() or (labels >= num_answers).any():
            print("Invalid labels detected:", labels)
            raise ValueError(f"Labels must be in range [0, {num_answers-1}]")

        with autocast(device_type=device.type):
            outputs = model(
                input_ids=encoding['input_ids'],
                token_type_ids=encoding['token_type_ids'],
                attention_mask=encoding['attention_mask'],
                visual_feats=visual_feats,
                visual_pos=visual_pos
            )
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        del questions, image_features, encoding, visual_feats, visual_pos, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Epoch {epoch}, Loss: {running_loss:.4f}, Accuracy: {correct / total:.4f}")


def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            try:
                # Extract batch data
                questions = batch['question']
                image_features = batch['image_features'].to(device)
                answers = batch['answers']

                # Tokenize questions
                encoding = tokenizer(
                    questions,
                    padding="max_length",
                    truncation=True,
                    max_length=20,
                    return_tensors="pt"
                ).to(device)

                # Project image features
                visual_feats = feature_projector(image_features.view(-1, 4)).view(
                    image_features.shape[0], image_features.shape[1], 2048
                )
                visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

                # Resolve answers to indices with <unk> as fallback
                resolved_answers = [
                    answer_to_index.get(resolve_answers(a), answer_to_index['<unk>'])
                    for a in answers
                ]
                labels = torch.tensor(resolved_answers, dtype=torch.long).to(device)

                # Verify label range
                if (labels < 0).any() or (labels >= num_answers).any():
                    print("Invalid labels detected in validation:", labels)
                    raise ValueError(f"Labels must be in range [0, {num_answers-1}]")

                # Forward pass
                with autocast(device_type=device.type):
                    outputs = model(
                        input_ids=encoding['input_ids'],
                        token_type_ids=encoding['token_type_ids'],
                        attention_mask=encoding['attention_mask'],
                        visual_feats=visual_feats,
                        visual_pos=visual_pos
                    )
                    loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            except Exception as e:
                print(f"Error in batch: {e}")
                continue

            # Free memory
            del questions, image_features, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

    if total > 0:  # Avoid division by zero
        print(f"Validation Loss: {running_loss:.4f}, Accuracy: {correct / total:.4f}")
    else:
        print("Validation skipped: No valid batches processed.")

    
# Main Training Loop
if __name__ == "__main__":
    for epoch in range(start_epoch, num_epochs + 1):
        train_one_epoch(epoch)
        validate()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch}")
    print("Training complete!")
