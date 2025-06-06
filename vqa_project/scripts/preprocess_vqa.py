import json
import pickle
from tqdm import tqdm

def process_vqa_data(questions_file, annotations_file, image_features_file, output_file):
    # Load questions
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    
    # Load preprocessed image features
    with open(image_features_file, 'rb') as f:
        image_features = pickle.load(f)
    
    # Combine data
    dataset = []
    skipped_entries = []
    print("Processing dataset...")
    for annotation in tqdm(annotations_data['annotations']):
        question_id = annotation['question_id']
        prefix = "COCO_train2014" if "train" in questions_file else "COCO_val2014"
        image_id = f"{prefix}_{str(annotation['image_id']).zfill(12)}.jpg" # Match image filenames
        answers = [ans['answer'] for ans in annotation['answers']]
        
        # Find the corresponding question
        question_data = next((q for q in questions_data['questions'] if q['question_id'] == question_id), None)
        if not question_data:
            skipped_entries.append({"type": "missing_question", "question_id": question_id, "image_id": image_id})
            continue
        question_text = question_data['question']
        
        # Get image features
        if image_id in image_features:
            image_feat = image_features[image_id]
        else:
            skipped_entries.append({"type": "missing_image_features", "question_id": question_id, "image_id": image_id})
            continue
        
        # Add to dataset
        dataset.append({
            'question_id': question_id,
            'image_id': image_id,
            'question': question_text,
            'answers': answers,
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
    train_questions_file = "D:/WORK/PG/Project/vqa_project/data/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
    train_annotations_file = "D:/WORK/PG/Project/vqa_project/data/vqa_v2/v2_mscoco_train2014_annotations.json"
    train_image_features_file = "D:/WORK/PG/Project/vqa_project/data/processed/image_features_train.pkl"
    train_output_file = "D:/WORK/PG/Project/vqa_project/data/processed/vqa_train_processed.pkl"
    
    val_questions_file = "D:/WORK/PG/Project/vqa_project/data/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json"
    val_annotations_file = "D:/WORK/PG/Project/vqa_project/data/vqa_v2/v2_mscoco_val2014_annotations.json"
    val_image_features_file = "D:/WORK/PG/Project/vqa_project/data/processed/image_features_val.pkl"
    val_output_file = "D:/WORK/PG/Project/vqa_project/data/processed/vqa_val_processed.pkl"
    
    # Process training data
    #process_vqa_data(train_questions_file, train_annotations_file, train_image_features_file, train_output_file)
    
    # Process validation data
    process_vqa_data(val_questions_file, val_annotations_file, val_image_features_file, val_output_file)
