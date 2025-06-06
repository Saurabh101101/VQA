import torch
from torch.amp import autocast
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import pickle
import os
from PIL import Image, UnidentifiedImageError
import gc


def extract_image_features(image_dir, output_file, transform=None, batch_size=16, chunk_size=10000, device="cuda"):
    # Load pre-trained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()

    # Ensure transformation is defined
    if transform is None:
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Split processing into chunks to handle large datasets
    image_files = os.listdir(image_dir)
    chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

    for chunk_idx, chunk_files in enumerate(chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}...")

        # Load intermediate results if available
        partial_output_file = f"{output_file}_chunk_{chunk_idx}.pkl"
        if os.path.exists(partial_output_file):
            print(f"Skipping chunk {chunk_idx + 1}, already processed.")
            continue

        features = {}

        # Process images in batches within the current chunk
        for i in range(0, len(chunk_files), batch_size):
            batch_files = chunk_files[i:i + batch_size]
            batch_tensors = []
            batch_filenames = []

            # Preprocess images
            for image_file in batch_files:
                image_path = os.path.join(image_dir, image_file)
                try:
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        batch_tensors.append(transform(img))
                        batch_filenames.append(image_file)
                except (UnidentifiedImageError, OSError) as e:
                    print(f"Skipping corrupted or invalid file: {image_file}. Error: {e}")
                    continue

            while True:  # Retry loop for OOM errors
                try:
                    batch_tensors = torch.stack(batch_tensors).to(device)

                    with torch.no_grad():
                        with autocast(device_type=device):
                            outputs = model(batch_tensors)

                    # Extract and store features
                    for filename, output in zip(batch_filenames, outputs):
                        features[filename] = output['boxes'].cpu().numpy()

                    break  # Exit retry loop on success
                except torch.cuda.OutOfMemoryError:
                    print(f"Out of memory for batch starting at index {i}. Retrying with smaller batch size...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    batch_tensors = batch_tensors[:batch_size]  # Retry with smaller batch

            print(f"Processed batch {i // batch_size + 1} of chunk {chunk_idx + 1} with batch size {batch_size}.")

        # Save partial results for the current chunk
        with open(partial_output_file, 'wb') as f:
            pickle.dump(features, f)
        print(f"Saved chunk {chunk_idx + 1} results to {partial_output_file}.")

        # Clean up memory after processing the chunk
        del features
        torch.cuda.empty_cache()
        gc.collect()

    # Merge all chunk files into the final output file
    final_features = {}
    for chunk_idx in range(len(chunks)):
        partial_output_file = f"{output_file}_chunk_{chunk_idx}.pkl"
        with open(partial_output_file, "rb") as f:
            chunk_features = pickle.load(f)
            final_features.update(chunk_features)

    with open(output_file, "wb") as f:
        pickle.dump(final_features, f)

    print(f"Saved all features to {output_file}.")


if __name__ == "__main__":
    image_dir = "D:/WORK/PG/Project/vqa_project/data/visual_genome/VG_100K_2"
    output_file = "vg_image_features_test.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_image_features(image_dir, output_file, batch_size=16, chunk_size=10000, device=device)
