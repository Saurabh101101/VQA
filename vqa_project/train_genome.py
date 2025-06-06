import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.lxmert_vg import LXMERTWithSceneGraph
from transformers import LxmertTokenizer
from torch.amp import GradScaler, autocast
import gc
import pickle
import json

from utils.data_loader import get_data_loaders

# Paths to preprocessed files
train_file = 'final_vg_train.pkl'
val_file = 'final_vg_test.pkl'
answer_mapping_file = "vg_answer_mapping.json"
relationships_file = 'cleaned_relationships.json'
attributes_file = 'cleaned_attributes.json'

# Configuration
batch_size = 8
num_epochs = 10
learning_rate = 1e-4
output_dir = "output/vg_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accumulation_steps = 4
checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize DataLoaders
train_loader, val_loader = get_data_loaders(
    train_file, val_file, relationships_file, attributes_file, batch_size=batch_size, num_workers=0
)

if os.path.exists(answer_mapping_file):
    with open(answer_mapping_file, 'r') as f:
        answer_to_index = json.load(f)
    num_answers = len(answer_to_index)
    print(f"Loaded answer mapping from {answer_mapping_file}")
else:
    if os.path.exists(train_file):
        with open(train_file, 'rb') as f:
            train_dataset = pickle.load(f)
        
        all_answers = []
        for sample in train_dataset:
            if 'answers' in sample:
                if isinstance(sample['answers'], list):
                    all_answers.extend(sample['answers'])
                else:
                    all_answers.append(sample['answers'])
        
        unique_answers = list(set(all_answers))
        answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}
        answer_to_index['<unk>'] = len(answer_to_index)
        num_answers = len(answer_to_index)
        
        with open(answer_mapping_file, 'w') as f:
            json.dump(answer_to_index, f, indent=4)
        print(f"Generated and saved answer mapping to {answer_mapping_file}")
    else:
        print(f"Error: {train_file} not found. Cannot generate answer mapping.")

# Model, Loss, Optimizer
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
model = LXMERTWithSceneGraph(num_answers=num_answers).to(device)
criterion = nn.CrossEntropyLoss()
feature_projector = nn.Linear(4, 2048).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
    """
    Resolves answers to the most frequent one in the batch and maps to an index.
    Handles nested lists gracefully.
    """
    if isinstance(answers[0], list):  # Nested list structure
        flat_answers = [item for sublist in answers for item in sublist]
    else:
        flat_answers = answers

    if flat_answers:
        most_frequent = max(set(flat_answers), key=flat_answers.count)
        return answer_to_index.get(most_frequent, answer_to_index['<unk>'])  # Map to index
    return answer_to_index['<unk>']  # Default for empty answers



def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        if batch is None:  # Skip empty batches
            continue

        questions = batch['question']
        image_features = torch.clamp(batch['image_features'].to(device), min=-1e6, max=1e6)
        scene_graph = batch['scene_graph']
        batch_indices = batch['batch'].to(device)
        answers = batch['answers']

        # Tokenize questions
        encoding = tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=20,
            return_tensors="pt"
        ).to(device)

        if image_features.shape[-1] != 2048:
            visual_feats = feature_projector(image_features.view(-1, image_features.shape[-1])).view(
                image_features.shape[0], image_features.shape[1], 2048
            )
        else:
            visual_feats = image_features

        # Clamp visual features
        visual_feats = torch.clamp(visual_feats, min=-1e6, max=1e6)
        visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

        # Extract and clamp scene graph components
        scene_nodes = torch.clamp(scene_graph['nodes'], min=-1e6, max=1e6).to(device)
        scene_edge_attr = torch.clamp(scene_graph['edge_attr'], min=-1e6, max=1e6).to(device)
        scene_edges = scene_graph['edges'].to(device)

        # Resolve answers to indices and clamp
        labels = torch.tensor([resolve_answers(a) for a in answers], dtype=torch.long).to(device)

        with autocast(device_type=device.type):
            outputs = model(
                input_ids=encoding['input_ids'],
                token_type_ids=encoding['token_type_ids'],
                attention_mask=encoding['attention_mask'],
                visual_feats=visual_feats,
                visual_pos=visual_pos,
                scene_nodes=scene_nodes,
                scene_edges=scene_edges,
                scene_edge_attr=scene_edge_attr,
                batch=batch_indices
            )
            
            # Handle problematic outputs gracefully
            outputs = torch.clamp(outputs, min=-1e6, max=1e6)
            outputs = outputs / torch.norm(outputs, p=2, dim=-1, keepdim=True).clamp(min=1e-6)

            # Detect invalid outputs and replace them with zeros
            # invalid_mask = torch.isnan(outputs).any(dim=-1) | torch.isinf(outputs).any(dim=-1)
            # if invalid_mask.any():
            #     print(f"Warning: Invalid outputs at step {step}. Correcting outputs.")
            #     outputs[invalid_mask] = torch.zeros_like(outputs[invalid_mask])

            # Replace NaN or Inf in outputs using nan_to_num
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)

            # Compute loss for all outputs, including corrected ones
            loss = criterion(outputs, labels) / accumulation_steps

        # Skip backward pass if loss is NaN or Inf
        if not torch.isfinite(loss):
            print(f"Skipping step {step} due to invalid loss value: {loss.item()}")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Cleanup unused variables
        del questions, image_features, scene_nodes, scene_edges, scene_edge_attr, encoding, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Epoch {epoch}, Loss: {running_loss:.4f}, Accuracy: {correct / total:.4f}")



# Validation Function
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if batch is None:  # Skip empty batches
                continue

            questions = batch['question']
            image_features = torch.clamp(batch['image_features'].to(device), min=-1e6, max=1e6)
            scene_graph = batch['scene_graph']
            batch_indices = batch['batch'].to(device)
            answers = batch['answers']

            # Tokenize questions
            encoding = tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                max_length=20,
                return_tensors="pt"
            ).to(device)

            # Process visual features
            if image_features.shape[-1] != 2048:
                visual_feats = feature_projector(image_features.view(-1, image_features.shape[-1])).view(
                    image_features.shape[0], image_features.shape[1], 2048
                )
            else:
                visual_feats = image_features

            visual_feats = torch.clamp(visual_feats, min=-1e6, max=1e6)
            visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)

            # Extract scene graph components
            scene_nodes = torch.clamp(scene_graph['nodes'], min=-1e6, max=1e6).to(device)
            scene_edge_attr = torch.clamp(scene_graph['edge_attr'], min=-1e6, max=1e6).to(device)
            scene_edges = scene_graph['edges'].to(device)

            # Resolve answers to indices and clamp labels
            labels = torch.tensor([resolve_answers(a) for a in answers], dtype=torch.long).to(device)

            # Correct invalid labels if detected
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print("Warning: Invalid labels detected. Correcting labels.")
                labels = torch.zeros_like(labels)

            with autocast(device_type=device.type):
                outputs = model(
                    input_ids=encoding['input_ids'],
                    token_type_ids=encoding['token_type_ids'],
                    attention_mask=encoding['attention_mask'],
                    visual_feats=visual_feats,
                    visual_pos=visual_pos,
                    scene_nodes=scene_nodes,
                    scene_edges=scene_edges,
                    scene_edge_attr=scene_edge_attr,
                    batch=batch_indices
                )

                # Clamp and normalize outputs
                outputs = torch.clamp(outputs, min=-1e6, max=1e6)
                outputs = outputs / torch.norm(outputs, p=2, dim=-1, keepdim=True).clamp(min=1e-6)

                # # Detect and correct invalid outputs
                # invalid_mask = torch.isnan(outputs).any(dim=-1) | torch.isinf(outputs).any(dim=-1)
                # if invalid_mask.any():
                #     print(f"Warning: Invalid outputs detected. Correcting outputs.")
                #     outputs[invalid_mask] = torch.zeros_like(outputs[invalid_mask])

                # Replace NaN or Inf in outputs using nan_to_num
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)

                # Compute loss
                loss = criterion(outputs, labels)

            # Update metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Cleanup
            del questions, image_features, scene_nodes, scene_edges, scene_edge_attr, encoding, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Validation Loss: {running_loss:.4f}, Accuracy: {correct / total:.4f}")






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
