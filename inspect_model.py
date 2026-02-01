import torch
import yaml
import os
import sys
from PIL import Image
import numpy as np
import timm

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.hybrid_model import create_model
from app.inference import load_model_from_checkpoint, predict_image

def inspect_model():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = 'models/alzheimer_model_final.pth'
    if not os.path.exists(model_path):
        return

    with open('model_info.txt', 'w') as out:
        out.write(f"Inspecting model: {model_path}\n")
        
        # 1. Create model and check initial weights
        model = create_model(config)
        first_conv_weight_before = model.backbone.conv_stem.weight.clone().detach()
        last_linear_weight_before = model.classifier[-1].weight.clone().detach()
        
        # 2. Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        
        first_conv_weight_after = model.backbone.conv_stem.weight.clone().detach()
        last_linear_weight_after = model.classifier[-1].weight.clone().detach()
        
        out.write("\n--- Weight Change Check ---\n")
        out.write(f"First Conv Weights Changed: {not torch.equal(first_conv_weight_before, first_conv_weight_after)}\n")
        out.write(f"Last Linear Weights Changed: {not torch.equal(last_linear_weight_before, last_linear_weight_after)}\n")
        
        out.write("\n--- Classifier Weights Statistics ---\n")
        out.write(f"Last Linear Weights Mean: {last_linear_weight_after.mean().item():.6f}\n")
        out.write(f"Last Linear Weights Std: {last_linear_weight_after.std().item():.6f}\n")
        
        # Check biases
        last_linear_bias = model.classifier[-1].bias.detach()
        out.write(f"Last Linear Biases: {last_linear_bias.numpy()}\n")
        
        # 3. Constant Input Tests
        out.write("\n--- Constant Input Tests ---\n")
        
        # Zero input
        zero_input = torch.zeros((1, 3, 224, 224))
        with torch.no_grad():
            output_zero = model(zero_input)
            prob_zero = torch.softmax(output_zero, dim=1).numpy()[0]
        out.write(f"Probabilities for Zero Input: {prob_zero}\n")
        
        # Random input
        rand_input = torch.randn((1, 3, 224, 224))
        with torch.no_grad():
            output_rand = model(rand_input)
            prob_rand = torch.softmax(output_rand, dim=1).numpy()[0]
        out.write(f"Probabilities for Random Input: {prob_rand}\n")
        
        # 4. Feature map check
        out.write("\n--- Architecture Check ---\n")
        features = model.backbone(zero_input)
        out.write(f"Number of feature stages: {len(features)}\n")
        for i, f in enumerate(features):
            out.write(f"Stage {i} shape: {f.shape}\n")
        
        x = model.attention(features[-1])
        out.write(f"After attention shape: {x.shape}\n")
        
if __name__ == "__main__":
    inspect_model()
