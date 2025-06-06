import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
def is_nan(value):
    """
    Check if a value is NaN (Not a Number).

    Args:
        value: The value to check.

    Returns:
        bool: True if the value is NaN, False otherwise.
    """
    try:
        return np.isnan(value)
    except TypeError:
        return False


def validate_scene_graph(scene_graph):
    """
    Validate the scene graph data to ensure 'nodes', 'edges', and 'edge_attr' contain valid, non-NaN values.

    Args:
        scene_graph (dict): A dictionary containing 'nodes', 'edges', and 'edge_attr' keys.

    Returns:
        dict: The cleaned and validated scene graph.
    """
    # Validate and clean 'nodes'
    if 'nodes' not in scene_graph or not isinstance(scene_graph['nodes'], torch.Tensor):
        raise ValueError("'nodes' must be a tensor in the scene graph.")
    if torch.isnan(scene_graph['nodes']).any():
        scene_graph['nodes'] = scene_graph['nodes'][~torch.isnan(scene_graph['nodes']).any(dim=1)]

    # Validate and clean 'edges'
    if 'edges' not in scene_graph or not isinstance(scene_graph['edges'], torch.Tensor):
        raise ValueError("'edges' must be a tensor in the scene graph.")
    if scene_graph['edges'].numel() == 0:
        raise ValueError("'edges' cannot be empty in the scene graph.")

    # Validate and clean 'edge_attr'
    if 'edge_attr' not in scene_graph or not isinstance(scene_graph['edge_attr'], torch.Tensor):
        raise ValueError("'edge_attr' must be a tensor in the scene graph.")
    if torch.isnan(scene_graph['edge_attr']).any():
        raise ValueError("'edge_attr' contains NaN values.")

    return scene_graph

class VGDataset(Dataset):
    def __init__(self, processed_file, relationships_file, attributes_file, batch_size, num_batches):
        self.processed_file = processed_file
        self.relationships_file = relationships_file
        self.attributes_file = attributes_file
        self.num_batches = num_batches
        self.batch_size = batch_size

        # Load the dataset
        with open(self.processed_file, 'rb') as f:
            self.data = pickle.load(f)
            if not self.data or len(self.data) < 1:
                raise ValueError(f"The dataset in {self.processed_file} is empty or malformed.")

        # Limit the data to the number of batches and batch size
        total_samples = self.num_batches * self.batch_size
        self.data = self.data[:total_samples]

        # Load relationships and attributes, ensuring alignment
        with open(relationships_file, 'r') as f:
            relationships_list = json.load(f)
        self.relationships = {str(item['image_id']): item['relationships'] for item in relationships_list}

        with open(attributes_file, 'r') as f:
            attributes_list = json.load(f)
        self.attributes = {str(item['image_id']): item['attributes'] for item in attributes_list}

        # Create predicate-to-index mapping
        self.predicate_to_index = {predicate: idx for idx, predicate in enumerate(
            set(rel['predicate'] for item in relationships_list for rel in item['relationships']))}

        # Validate alignment
        dataset_ids = {str(sample['image_id']) for sample in self.data}
        relationships_ids = set(self.relationships.keys())
        attributes_ids = set(self.attributes.keys())
        missing_ids = dataset_ids - (relationships_ids & attributes_ids)
        if missing_ids:
            raise ValueError(f"Scene graph data missing for image IDs: {missing_ids}")

        # Sequential indices for processing
        self.indices = list(range(len(self.data)))
        

    def process_scene_graph(self, image_id):
        """
        Convert JSON relationships and attributes into tensors for GNN processing.
        """
        image_id_str = str(image_id)  # Ensure image_id is treated as a string key

        # Extract relationships and attributes
        relationships = self.relationships[image_id_str]
        attributes = self.attributes[image_id_str]

        # Define a fixed vector size for node features
        default_vector_length = 5
        default_vector = [0.0] * default_vector_length

        # Extract nodes
        nodes = []
        node_id_map = {}
        for i, obj in enumerate(attributes):
            attr = obj.get('attributes', [])
            if isinstance(attr, list) and all(isinstance(x, (int, float)) for x in attr):
                attr = attr[:default_vector_length] + [0.0] * (default_vector_length - len(attr))
            else:
                attr = default_vector
            nodes.append(attr)
            node_id_map[obj['object_id']] = i

        nodes = torch.tensor(nodes, dtype=torch.float32)

        # Extract edges and edge attributes
        edges = []
        edge_attr = []
        for rel in relationships:
            subject_id = rel['subject']['object_id']
            object_id = rel['object']['object_id']
            if subject_id in node_id_map and object_id in node_id_map:
                edges.append([node_id_map[subject_id], node_id_map[object_id]])
                edge_attr.append(self.predicate_to_index[rel['predicate']])

        # Skip images with no valid edges
        if not edges:
            raise ValueError(f"No edges found for image_id: {image_id}")

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)  # Convert to tensor

        # Create the scene graph dictionary
        scene_graph = {
            "nodes": nodes,
            "edges": edge_index,
            "edge_attr": edge_attr
        }

        # Validate the scene graph
        try:
            scene_graph = validate_scene_graph(scene_graph)
        except ValueError as e:
            raise ValueError(f"Validation failed for scene graph of image_id {image_id}: {e}")

        return scene_graph['nodes'], scene_graph['edges'], scene_graph['edge_attr']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.data[actual_idx]
        image_id = sample['image_id']

        try:
            # Process scene graph
            scene_nodes, scene_edges, scene_edge_attr = self.process_scene_graph(image_id)
        except ValueError as e:
            print(f"Skipping image_id {image_id}: {e}")
            return None  # Move to the next index

        question = sample['question']
        image_features = torch.tensor(sample['image_features'], dtype=torch.float32)
        answers = sample.get('answer', '<unk>')  # Use '<unk>' as a default if the key is missing

        return {
            'question': question,
            'image_features': image_features,
            'scene_graph': {
                'nodes': scene_nodes,
                'edges': scene_edges,
                'edge_attr': scene_edge_attr
            },
            'answers': answers
        }


def custom_collate_fn(batch):
    # Filter out invalid samples (e.g., skipped images)
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:  # Skip if the batch becomes empty
        print("Empty batch after filtering invalid samples. Skipping batch.")
        return None
    
    questions = [item['question'] for item in batch]
    answers = [item['answers'] for item in batch]

    # Pad image features
    max_len = max(item['image_features'].shape[0] for item in batch)
    padded_image_features = [
        torch.nn.functional.pad(
            item['image_features'],
            (0, 0, 0, max_len - item['image_features'].shape[0])
        ) for item in batch
    ]
    padded_image_features = torch.stack(padded_image_features, dim=0)

    # Batch scene graph nodes
    scene_nodes = torch.cat([item['scene_graph']['nodes'] for item in batch], dim=0)

    # Adjust edge indices for batching
    node_offsets = torch.cumsum(
        torch.tensor([0] + [item['scene_graph']['nodes'].shape[0] for item in batch[:-1]]),
        dim=0
    )
    scene_edges = torch.cat([
        item['scene_graph']['edges'] + offset
        for item, offset in zip(batch, node_offsets)
    ], dim=1)
    scene_edge_attr = torch.cat([item['scene_graph']['edge_attr'] for item in batch], dim=0)
    # Create batch tensor for pooling
    batch_indices = torch.cat([
        torch.full((item['scene_graph']['nodes'].shape[0],), i, dtype=torch.long)
        for i, item in enumerate(batch)
    ])
    
    # Concatenate edge attributes
    

    return {
        'question': questions,
        'image_features': padded_image_features,
        'scene_graph': {
            'nodes': scene_nodes,
            'edges': scene_edges,
            'edge_attr': scene_edge_attr,
        },
        'batch': batch_indices,
        'answers': answers
    }



def get_data_loaders(train_file, test_file, relationships_file, attributes_file, batch_size=8, num_workers=0):
    """
    Create data loaders for training and validation datasets.
    Limits the number of batches per epoch using sequential processing.
    """
    train_dataset = VGDataset(train_file, relationships_file, attributes_file, batch_size=batch_size, num_batches=200)
    test_dataset = VGDataset(test_file, relationships_file, attributes_file, batch_size=batch_size, num_batches=100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Ensure sequential processing
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Ensure sequential processing
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return train_loader, test_loader
