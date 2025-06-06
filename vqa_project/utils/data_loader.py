import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class VQADataset(Dataset):
    def __init__(self, processed_file):
        self.processed_file = processed_file

        # Load the entire dataset once
        with open(self.processed_file, 'rb') as f:
            self.data = pickle.load(f)
            if not self.data or len(self.data) < 1:
                raise ValueError(f"The dataset in {self.processed_file} is empty or malformed.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample['question']
        image_features = torch.tensor(sample['image_features'], dtype=torch.float32)
        answers = sample['answers']

        return {
            'question': question,
            'image_features': image_features,
            'answers': answers
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors.
    Pads image features to the maximum sequence length in the batch.
    """
    questions = [item['question'] for item in batch]
    answers = [item['answers'] for item in batch]

    # Find the maximum sequence length for image features in the batch
    max_len = max(item['image_features'].shape[0] for item in batch)

    # Pad image features to the maximum length
    padded_image_features = [
        torch.nn.functional.pad(
            item['image_features'],
            (0, 0, 0, max_len - item['image_features'].shape[0])  # Pad along the first dimension
        ) for item in batch
    ]
    padded_image_features = torch.stack(padded_image_features, dim=0)

    return {
        'question': questions,
        'image_features': padded_image_features,
        'answers': answers
    }


def get_data_loaders(train_file=None, val_file=None, batch_size=8, num_workers=0):
    train_loader = None

    if train_file:  # Only load if train_file is provided
        try:
            train_dataset = VQADataset(train_file)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)
        except ValueError as e:
            print(f"Warning: Skipping training dataset due to error - {e}")

    val_loader = None
    if val_file:
        val_dataset = VQADataset(val_file)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)

    return train_loader, val_loader


# def get_data_loaders(train_file, val_file, batch_size=8, num_workers=0):
#     """
#     Create data loaders for training and validation datasets.
#     Limits the number of batches per epoch using random sampling.
#     """
#     train_dataset = VQADataset(train_file, batch_size=batch_size, num_batches=13867)
#     val_dataset = VQADataset(val_file, batch_size=batch_size, num_batches=6698)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=False,  # Random sampling is handled in VQADataset
#         num_workers=num_workers,
#         collate_fn=custom_collate_fn
#         # persistent_workers=True if num_workers > 0 else False,
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         collate_fn=custom_collate_fn
#         # persistent_workers=True if num_workers > 0 else False,
#     )

#     return train_loader, val_loader
