import torch
import yaml
import os
import sys
from PIL import Image
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.hybrid_model import create_model
from app.inference import load_model_from_checkpoint, predict_image

def evaluate_on_subset():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = r'notebooks\data\raw\Data'
    class_folders = {
        'NonDemented': 'Non Demented',
        'VeryMildDemented': 'Very mild Dementia',
        'MildDemented': 'Mild Dementia',
        'ModerateDemented': 'Moderate Dementia'
    }

    model_path = 'models/alzheimer_model_final.pth'
    if not os.path.exists(model_path):
        print("Model file not found")
        return

    model, device = load_model_from_checkpoint(model_path, config)
    model.eval()
    
    class_names = config['data']['class_names']
    results = {}

    for canonical, folder in class_folders.items():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print(f"No images in {folder_path}")
            continue
            
        # Take up to 20 images
        files = files[:20]
        preds = []
        
        for f in files:
            img_path = os.path.join(folder_path, f)
            img = Image.open(img_path).convert('RGB')
            prediction, probabilities = predict_image(img, model, device, config)
            preds.append(prediction)
            
        results[canonical] = preds

    print("\n--- Evaluation Results (Index Counts) ---")
    for canonical, preds in results.items():
        counts = {i: preds.count(i) for i in range(4)}
        print(f"Ground Truth: {canonical}")
        for i, name in enumerate(class_names):
            print(f"  Predicted {name}: {counts[i]}")

if __name__ == "__main__":
    evaluate_on_subset()
