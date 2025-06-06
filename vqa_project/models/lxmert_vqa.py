# import torch
# import torch.nn as nn
# from transformers import LxmertModel, LxmertTokenizer
# from torch_geometric.nn import GCNConv

# class LXMERTForVQA(nn.Module):
#     def __init__(self, num_answers):
#         super(LXMERTForVQA, self).__init__()
        
#         # Load the LXMERT model
#         self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#         self.classifier = nn.Linear(self.lxmert.config.hidden_size, num_answers)
#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(self.lxmert.config.hidden_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, num_answers)
#         )

#     def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
#         # Pass inputs through LXMERT
#         outputs = self.lxmert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             visual_feats=visual_feats,
#             visual_pos=visual_pos
#         )

#         # Extract the pooled output
#         # pooled_output = outputs[1]  # [CLS] token representation
#         pooled_output = outputs.pooled_output
#         # Pass through the classifier
#         logits = self.classifier(pooled_output)

#         return logits


import torch
import torch.nn as nn
from transformers import LxmertModel, LxmertTokenizer
from torch_geometric.nn import GCNConv

class LXMERTForVQA(nn.Module):
    def __init__(self, num_answers):
        super(LXMERTForVQA, self).__init__()
        
        # Load the LXMERT model
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.classifier = nn.Linear(self.lxmert.config.hidden_size, num_answers)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.lxmert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_answers)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos):
        # Pass inputs through LXMERT
        outputs = self.lxmert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos
        )

        # Extract the pooled output
        # pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = outputs.pooled_output
        # Pass through the classifier
        logits = self.classifier(pooled_output)

        return logits
