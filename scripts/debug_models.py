import torch
import yaml
import os
from PIL import Image
import numpy as np
import sys

# Add project root
sys.path.append(os.path.abspath(os.curdir))

from app.inference import load_model_from_checkpoint, predict_image

def debug_models():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"Current Config Class Names: {config['data']['class_names']}")
    
    moderate_dir = r'notebooks\data\raw\Data\Moderate Dementia'
    if not os.path.exists(moderate_dir):
        print("Moderate directory not found")
        return

    files = [f for f in os.listdir(moderate_dir) if f.lower().endswith(('.jpg', '.png'))]
    img_path = os.path.join(moderate_dir, files[0])
    image = Image.open(img_path).convert('RGB')
    
    archs = ['cnn', 'vit', 'hybrid']
    for arch in archs:
        print(f"\n--- {arch.upper()} ---")
        checkpoint_path = config['paths'].get(f'{arch}_model')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print("No checkpoint")
            continue
        
        try:
            model, device = load_model_from_checkpoint(checkpoint_path, config, arch)
            pred, probs = predict_image(image, model, device, config)
            print(f"Predicted Index: {pred}")
            print(f"Probabilities: {probs}")
            print(f"Mapped Name: {config['data']['class_names'][pred]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_models()
