# import os
# import torch
# import torch.nn as nn
# import pickle
# from transformers import LxmertTokenizer
# from models.lxmert_vqa import LXMERTForVQA
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import json

# # Paths and configurations
# checkpoint_path = "D:/WORK/PG/Project/vqa_final/vqa_model.pth"
# answer_mapping_path = "D:/WORK/PG/Project/vqa_final/vqa_answer_mapping.json"  # Path to pre-existing answer mapping file
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load pre-existing answer mapping
# with open(answer_mapping_path, 'rb') as f:
#     answer_to_index = json.load(f)
#     index_to_answer = {idx: answer for answer, idx in answer_to_index.items()}

# # Load tokenizer
# tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# # Load trained model
# num_answers = len(answer_to_index)
# model = LXMERTForVQA(num_answers=num_answers).to(device)
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# # Feature projector
# feature_projector = nn.Linear(4, 2048).to(device)

# def preprocess_image(image_path):
#     """Preprocess the image and return the image features."""
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(device)
    
#     # Simulated image features (4D bounding box features as a placeholder)
#     image_features = torch.randn(1, 36, 4).to(device)  # Example: 36 regions with 4D features
#     visual_feats = feature_projector(image_features.view(-1, 4)).view(image_features.shape[0], image_features.shape[1], 2048)
#     visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)
    
#     return visual_feats, visual_pos

# def predict_answer(image_path, question):
#     """Given an image and a question, return the predicted answer with index."""
#     visual_feats, visual_pos = preprocess_image(image_path)

#     encoding = tokenizer(
#         question,
#         padding="max_length",
#         truncation=True,
#         max_length=20,
#         return_tensors="pt"
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(
#             input_ids=encoding['input_ids'],
#             token_type_ids=encoding['token_type_ids'],
#             attention_mask=encoding['attention_mask'],
#             visual_feats=visual_feats,
#             visual_pos=visual_pos
#         )

#         # Ensure outputs is of shape [1, num_answers]
#         if outputs.dim() > 2:
#             outputs = outputs.squeeze(0)

#         predicted_idx = torch.argmax(outputs, dim=1).item()
#         predicted_answer = index_to_answer.get(predicted_idx, "<unknown>")

#     return predicted_idx, predicted_answer

# # Example usage
# if __name__ == "__main__":
#     image_path = "6.jpg"  # Replace with actual image path
#     question = "What colour is the snow?"  # Replace with your question
#     predicted_idx, prediction = predict_answer(image_path, question)
#     print(f"Question: {question}")
#     print(f"Predicted Answer: {prediction}")


# import os
# import torch
# from transformers import LxmertTokenizer
# from models.lxmert_vqa import LXMERTForVQA
# import pickle
# import torch.nn as nn

# # Paths
# output_dir = "output/vqa_model"
# checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
# test_file = "vqa_train.pkl"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feature_projector = nn.Linear(4, 2048).to(device)

# def load_model():
#     # Load answer mapping
#     with open("vqa_train.pkl", 'rb') as f:
#         train_dataset = pickle.load(f)
#     all_answers = [answer for sample in train_dataset for answer in sample['answers']]
#     unique_answers = list(set(all_answers))
#     answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}
#     index_to_answer = {idx: answer for answer, idx in answer_to_index.items()}
#     num_answers = len(unique_answers)
    
#     # Load model
#     model = LXMERTForVQA(num_answers=num_answers).to(device)
#     tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
#         print("Model loaded successfully!")
#     else:
#         raise FileNotFoundError("Checkpoint file not found!")
    
#     return model, tokenizer, index_to_answer

# def predict_from_pkl(model, tokenizer, index_to_answer, test_file):
#     with open(test_file, 'rb') as f:
#         test_data = pickle.load(f)
    
#     for sample in test_data:
#         question = sample['question']
#         image_features = torch.tensor(sample['image_features']).unsqueeze(0).to(device)
#         # visual_pos = torch.zeros(1, image_features.shape[1], 4).to(device)
        
#         encoding = tokenizer(
#             question,
#             padding="max_length",
#             truncation=True,
#             max_length=20,
#             return_tensors="pt"
#         ).to(device)
        
#         visual_feats = feature_projector(image_features.view(-1, 4)).view(
#             image_features.shape[0], image_features.shape[1], 2048
#         )
#         visual_pos = torch.zeros(image_features.shape[0], image_features.shape[1], 4).to(device)
        
#         with torch.no_grad():
#             outputs = model(
#                 input_ids=encoding['input_ids'],
#                 token_type_ids=encoding['token_type_ids'],
#                 attention_mask=encoding['attention_mask'],
#                 visual_feats=visual_feats,
#                 visual_pos=visual_pos
#             )
        
#         predicted_index = torch.argmax(outputs, dim=1).item()
#         predicted_answer = index_to_answer.get(predicted_index, "Unknown")
#         print(f"Question: {question}\nPredicted Answer: {predicted_answer}\n")

# if __name__ == "__main__":
#     model, tokenizer, index_to_answer = load_model()
#     predict_from_pkl(model, tokenizer, index_to_answer, test_file)

# import os
# import torch
# import torch.nn as nn
# import pickle
# from transformers import LxmertTokenizer
# from models.lxmert_vqa import LXMERTForVQA
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import json
# import torchvision
# from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights




# # Paths and configurations
# checkpoint_path = "D:/WORK/PG/Project/vqa_final/vqa_model.pth"
# answer_mapping_path = "D:/WORK/PG/Project/vqa_final/vqa_answer_mapping.json"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load pre-existing answer mapping
# with open(answer_mapping_path, 'rb') as f:
#     answer_to_index = json.load(f)
#     index_to_answer = {idx: answer for answer, idx in answer_to_index.items()}

# # Load tokenizer
# tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# # Load trained model
# num_answers = len(answer_to_index)
# model = LXMERTForVQA(num_answers=num_answers).to(device)
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()


# # Load Faster R-CNN with COCO weights
# weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
# feature_extractor = fasterrcnn_resnet50_fpn(weights=weights).to(device)
# feature_extractor.eval()

# def extract_image_features(image_path):
#     """Extracts object-level image features using Faster R-CNN."""
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(device)  # Move image to the same device

#     with torch.no_grad():
#         detections = feature_extractor(image_tensor)

#     if "boxes" in detections[0] and "scores" in detections[0]:
#         boxes = detections[0]["boxes"]
#         scores = detections[0]["scores"]
        
#         # Select top-k highest confidence detections (max 36 objects)
#         num_objects = min(36, len(boxes))  
#         top_k_indices = scores.topk(num_objects).indices  
        
#         selected_boxes = boxes[top_k_indices].to(device)
        
#         # Extract feature maps from backbone (ResNet)
#         feature_maps = feature_extractor.backbone(image_tensor)  # Returns dict of feature maps
        
#         # Use ROI Align to extract object features from feature maps
#         pooled_features = torchvision.ops.roi_align(
#             feature_maps['0'],  # Select feature map level
#             [selected_boxes],  # Bounding boxes
#             output_size=(7, 7),  # Pooling output size
#             spatial_scale=1.0
#         )

#         visual_feats = pooled_features.view(num_objects, -1)  # Flatten ROI-pooled features

#         # Ensure correct feature shape (num_objects, 2048)
#         if visual_feats.shape[1] != 2048:
#             print(f"Warning: Expected (num_objects, 2048), but got {visual_feats.shape}")
#             return None, None  
        
#         return visual_feats.unsqueeze(0), selected_boxes.unsqueeze(0)

#     return None, None  # No objects detected

# def predict_answer(image_path, question):
#     """Predict answer based on the given image and question."""
#     visual_feats, visual_pos = extract_image_features(image_path)

#     # If no valid features are extracted, return a fallback response
#     if visual_feats is None or visual_pos is None:
#         return -1, "<unknown>"  # Model cannot answer without image context

#     encoding = tokenizer(
#         question,
#         padding="max_length",
#         truncation=True,
#         max_length=20,
#         return_tensors="pt"
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(
#             input_ids=encoding['input_ids'],
#             token_type_ids=encoding['token_type_ids'],
#             attention_mask=encoding['attention_mask'],
#             visual_feats=visual_feats,
#             visual_pos=visual_pos
#         )

#         if outputs.dim() > 2:
#             outputs = outputs.squeeze(0)

#         predicted_idx = torch.argmax(outputs, dim=1).item()
#         predicted_answer = index_to_answer.get(predicted_idx, "<unknown>")

#     return predicted_idx, predicted_answer

# # Example usage
# if __name__ == "__main__":
#     image_path = "6.jpg"  # Replace with actual image path
#     question = "What colour is the snow?"
#     predicted_idx, prediction = predict_answer(image_path, question)
#     print(f"Predicted Index: {predicted_idx}, Predicted Answer: {prediction}")

import os
import torch
import torch.nn as nn
import pickle
from transformers import LxmertTokenizer
from models.lxmert_vqa import LXMERTForVQA
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Paths and configurations
checkpoint_path = "D:/WORK/PG/Project/vqa_final/combined_model.pth"  # Use combined model
answer_mapping_path = "D:/WORK/PG/Project/vqa_final/vqa_answer_mapping.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-existing answer mapping
with open(answer_mapping_path, 'rb') as f:
    answer_to_index = json.load(f)
index_to_answer = {idx: answer for answer, idx in answer_to_index.items()}

# Load tokenizer
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# Load model parameters
num_answers = len(answer_to_index)

# Define Combined Model
class CombinedLXMERT(nn.Module):
    def __init__(self, model_vqa, model_vg):
        super(CombinedLXMERT, self).__init__()
        self.model_vqa = model_vqa
        self.model_vg = model_vg

    def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
        logits_vqa = self.model_vqa(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)
        logits_vg = self.model_vg(input_ids, attention_mask, token_type_ids, visual_feats, visual_pos)

        # Compute confidence scores
        confidence_vqa = torch.max(torch.softmax(logits_vqa, dim=1), dim=1)[0]
        confidence_vg = torch.max(torch.softmax(logits_vg, dim=1), dim=1)[0]

        # Select the model with higher confidence
        combined_logits = torch.where(confidence_vqa.unsqueeze(1) > confidence_vg.unsqueeze(1), logits_vqa, logits_vg)

        return combined_logits

# Load base models
model_vqa = LXMERTForVQA(num_answers=num_answers).to("cpu")
model_vg = LXMERTForVQA(num_answers=num_answers).to("cpu")

# Load combined model
combined_model = CombinedLXMERT(model_vqa, model_vg)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
combined_model.load_state_dict(checkpoint)
combined_model.to(device)
combined_model.eval()

print("Combined model loaded successfully.")

# Load Faster R-CNN with COCO weights
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
feature_extractor = fasterrcnn_resnet50_fpn(weights=weights).to(device)
feature_extractor.eval()

def extract_image_features(image_path):
    """Extracts object-level image features using Faster R-CNN."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = feature_extractor(image_tensor)

    if "boxes" in detections[0] and "scores" in detections[0]:
        boxes = detections[0]["boxes"]
        scores = detections[0]["scores"]
        
        # Select top-k highest confidence detections (max 36 objects)
        num_objects = min(36, len(boxes))  
        top_k_indices = scores.topk(num_objects).indices  
        
        selected_boxes = boxes[top_k_indices].to(device)
        
        # Extract feature maps from backbone (ResNet)
        feature_maps = feature_extractor.backbone(image_tensor)

        # Use ROI Align to extract object features from feature maps
        pooled_features = torchvision.ops.roi_align(
            feature_maps['0'],  # Select feature map level
            [selected_boxes],  # Bounding boxes
            output_size=(7, 7),  # Pooling output size
            spatial_scale=1.0
        )

        visual_feats = pooled_features.view(num_objects, -1)  # Flatten ROI-pooled features

        # Ensure correct feature shape (num_objects, 2048)
        if visual_feats.shape[1] != 2048:
            print(f"Warning: Expected (num_objects, 2048), but got {visual_feats.shape}")
            return None, None  
        
        return visual_feats.unsqueeze(0), selected_boxes.unsqueeze(0)

    return None, None  # No objects detected

def predict_answer(image_path, question):
    """Predict answer based on the given image and question using the combined model."""
    visual_feats, visual_pos = extract_image_features(image_path)

    # If no valid features are extracted, return a fallback response
    if visual_feats is None or visual_pos is None:
        return -1, "<unknown>"  # Model cannot answer without image context

    encoding = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=20,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = combined_model(
            input_ids=encoding['input_ids'],
            token_type_ids=encoding['token_type_ids'],
            attention_mask=encoding['attention_mask'],
            visual_feats=visual_feats,
            visual_pos=visual_pos
        )

        if outputs.dim() > 2:
            outputs = outputs.squeeze(0)

        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_answer = index_to_answer.get(predicted_idx, "<unknown>")

    return predicted_idx, predicted_answer

# Example usage
if __name__ == "__main__":
    image_path = "6.jpg"  # Replace with actual image path
    question = "What colour is the snow?"
    predicted_idx, prediction = predict_answer(image_path, question)
    print(f"Predicted Index: {predicted_idx}, Predicted Answer: {prediction}")
