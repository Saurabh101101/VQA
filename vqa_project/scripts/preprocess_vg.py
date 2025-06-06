import json
import pickle
from tqdm import tqdm

def process_visual_genome_data(qna_file, image_features_file, output_file):
    # Load QnA data
    with open(qna_file, 'r') as f:
        qna_data = json.load(f)
    
    # Load preprocessed image features
    with open(image_features_file, 'rb') as f:
        image_features = pickle.load(f)
    
    # Combine data
    dataset = []
    skipped_entries = []
    print("Processing Visual Genome dataset...")
    
    for image_entry in tqdm(qna_data):
        image_id = image_entry['id']  # Image ID from the JSON
        image_key = f"{image_id}.jpg"  # Match key in the image features file
        
        # Get image features
        if image_key in image_features:
            image_feat = image_features[image_key]
        else:
            skipped_entries.append({"type": "missing_image_features", "image_id": image_id, "image_key": image_key})
            continue
        
        # Process QA pairs for this image
        for qa in image_entry['qas']:
            question = qa['question']
            answer = qa['answer']  # Single answer as per provided structure
            qa_id = qa['qa_id']
            
            # Add to dataset
            dataset.append({
                'qa_id': qa_id,
                'image_id': image_id,
                'question': question,
                'answer': answer,
                'image_features': image_feat
            })
    
    # Save processed dataset
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved processed dataset to {output_file} with {len(dataset)} entries.")

    # Save skipped entries for debugging
    skipped_file = output_file.replace(".pkl", "_skipped.json")
    with open(skipped_file, 'w') as f:
        json.dump(skipped_entries, f, indent=4)
    print(f"Saved skipped entries log to {skipped_file}.")

if __name__ == "__main__":
    # Paths to files
    qna_file = "D:/WORK/PG/Project/vqa_project/data/visual_genome/question_answers.json"
    image_features_file = "D:/WORK/PG/Project/vqa_project/data/processed/vg_image_features_test.pkl"
    output_file = "D:/WORK/PG/Project/vqa_project/data/processed/vg_test_processed.pkl"
    
    # Process Visual Genome data
    process_visual_genome_data(qna_file, image_features_file, output_file)
