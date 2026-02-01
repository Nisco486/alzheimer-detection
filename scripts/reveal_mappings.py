import torch
import yaml
import os
from PIL import Image
import numpy as np
import sys

# Add project root
sys.path.append(os.path.abspath(os.curdir))

from app.inference import load_model_from_checkpoint, predict_image

def reveal_mappings():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    root_dir = r'notebooks\data\raw\Data'
    classes = [
        ('Mild Dementia', 'MildDemented'),
        ('Moderate Dementia', 'ModerateDemented'),
        ('Non Demented', 'NonDemented'),
        ('Very mild Dementia', 'VeryMildDemented')
    ]
    
    archs = ['cnn', 'vit', 'hybrid']
    
    results = {}
    
    for arch in archs:
        results[arch] = {}
        checkpoint_path = config['paths'].get(f'{arch}_model')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            continue
        
        model, device = load_model_from_checkpoint(checkpoint_path, config, arch)
        
        for folder, label in classes:
            folder_path = os.path.join(root_dir, folder)
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
            img_path = os.path.join(folder_path, files[0])
            image = Image.open(img_path).convert('RGB')
            
            pred, probs = predict_image(image, model, device, config)
            results[arch][label] = pred

    for arch in archs:
        print(f"\nModel: {arch}")
        for folder, label in classes:
            pred = results[arch].get(label, 'N/A')
            print(f"  {label:20} -> {pred}")

if __name__ == "__main__":
    reveal_mappings()
