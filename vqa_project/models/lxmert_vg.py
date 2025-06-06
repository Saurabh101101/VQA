import torch
import torch.nn as nn
from transformers import LxmertModel
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GINEConv

class LXMERTWithSceneGraph(nn.Module):
    def __init__(self, num_answers):
        super(LXMERTWithSceneGraph, self).__init__()
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

        # Linear projection for node and edge features
        self.node_proj = nn.Linear(5, 2048)
        self.edge_proj = nn.Linear(1, 2048)  # Adjust input dim if edge features are not scalar

        # Scene Graph GNN
        self.gcn1 = GINEConv(
            nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024)
            ),
            edge_dim=2048
        )
        self.gcn2 = GINEConv(
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            ),
            edge_dim=2048
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.lxmert.config.hidden_size + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_answers)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos, scene_nodes, scene_edges, scene_edge_attr, batch):
        # Move tensors to the correct device
        device = next(self.parameters()).device
        scene_nodes = scene_nodes.to(device)
        scene_edges = scene_edges.to(device)
        scene_edge_attr = scene_edge_attr.to(device)

        # LXMERT Outputs
        lxmert_outputs = self.lxmert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos
        )
        pooled_output = lxmert_outputs.pooled_output

        # Transform node and edge features
        scene_nodes = self.node_proj(scene_nodes)
        scene_edge_attr = self.edge_proj(scene_edge_attr.unsqueeze(1).float())  # Ensure float type
  # Project edge attributes

        # Process scene graph with GNN
        scene_features = self.gcn1(scene_nodes, scene_edges, edge_attr=scene_edge_attr)
        scene_features = self.gcn2(scene_features, scene_edges, edge_attr=scene_edge_attr)

        # Aggregate scene graph features
        scene_graph_representation = global_mean_pool(scene_features, batch)

        # Combine LXMERT and scene graph features
        combined_features = torch.cat([pooled_output, scene_graph_representation], dim=1)
        logits = self.classifier(combined_features)

        return logits
