import torch
import numpy as np
from PIL import Image
# import albumentations as A  # Moved inside functions to avoid Streamlit/atexit issues
# from albumentations.pytorch import ToTensorV2

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_model import create_model as create_hybrid_model
from models.cnn_backbone import create_cnn_model
from models.vision_transformer import create_vit_model

def get_inference_transform(img_size=224):
    """Get transformation for inference"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    return transform

def load_model_from_checkpoint(checkpoint_path, config, architecture='hybrid'):
    """Load trained model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model based on architecture
    if architecture == 'hybrid':
        model = create_hybrid_model(config)
    elif architecture == 'cnn':
        model = create_cnn_model(config)
    elif architecture == 'vit':
        model = create_vit_model(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load checkpoint if provided and valid
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check if it's the state_dict directly or a dictionary containing it
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Handle potential mismatches (strict=False) since we might use different archs
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load weights for {architecture}: {e}")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def preprocess_image(image, config):
    """Preprocess image for model input"""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    # Apply transformations
    transform = get_inference_transform(config['data']['img_size'])
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_image(image, model, device, config):
    """
    Predict Alzheimer's stage from MRI image
    
    Args:
        image: PIL Image or numpy array
        model: Trained model
        device: torch device
        config: Configuration dictionary
    
    Returns:
        prediction: Predicted class index
        probabilities: Probability distribution over classes
    """
    # Preprocess image
    image_tensor = preprocess_image(image, config)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    # Convert to numpy
    prediction = prediction.cpu().item()
    probabilities = probabilities.cpu().numpy()[0]
    
    return prediction, probabilities

def predict_batch(images, model, device, config):
    """
    Predict multiple images at once
    
    Args:
        images: List of PIL Images or numpy arrays
        model: Trained model
        device: torch device
        config: Configuration dictionary
    
    Returns:
        predictions: List of predicted class indices
        probabilities: Array of probability distributions
    """
    # Preprocess all images
    image_tensors = []
    for image in images:
        tensor = preprocess_image(image, config)
        image_tensors.append(tensor)
    
    # Stack into batch
    batch_tensor = torch.cat(image_tensors, dim=0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    # Convert to numpy
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    return predictions, probabilities

def get_class_name(prediction, config):
    """Get class name from prediction index"""
    return config['data']['class_names'][prediction]

def get_confidence_level(probability):
    """Categorize confidence level"""
    if probability >= 0.9:
        return "Very High"
    elif probability >= 0.75:
        return "High"
    elif probability >= 0.6:
        return "Moderate"
    elif probability >= 0.5:
        return "Low"
    else:
        return "Very Low"

if __name__ == "__main__":
    # Test inference
    import yaml
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    checkpoint_path = config['paths']['best_model']
    model, device = load_model_from_checkpoint(checkpoint_path, config)
    
    # Test with dummy image
    dummy_image = Image.new('RGB', (224, 224), color='gray')
    prediction, probabilities = predict_image(dummy_image, model, device, config)
    
    print(f"Prediction: {get_class_name(prediction, config)}")
    print(f"Probabilities: {probabilities}")
    print(f"Confidence: {get_confidence_level(probabilities[prediction])}")