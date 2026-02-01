import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.hybrid_model import create_model
from app.inference import load_model_from_checkpoint

class SmallDataset(Dataset):
    def __init__(self, data_root, class_folders, transform=None, samples_per_class=100):
        self.samples = []
        self.transform = transform
        
        for idx, (canonical, folder) in enumerate(class_folders.items()):
            folder_path = os.path.join(data_root, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                # Limit samples
                files = files[:samples_per_class]
                for f in files:
                    self.samples.append((os.path.join(folder_path, f), idx))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        if self.transform:
            img_np = self.transform(image=img_np)['image']
        return img_np, label

def fine_tune_classifier():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_root = r'notebooks\data\raw\Data'
    class_folders = {
        'NonDemented': 'Non Demented',
        'VeryMildDemented': 'Very mild Dementia',
        'MildDemented': 'Mild Dementia',
        'ModerateDemented': 'Moderate Dementia'
    }

    model_path = 'models/alzheimer_model_final.pth'
    model, device = load_model_from_checkpoint(model_path, config)
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.attention.parameters():
        param.requires_grad = False
        
    # We will train the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Take 100 images per class for a fast test
    dataset = SmallDataset(data_root, class_folders, transform=transform, samples_per_class=100)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Starting fine-tuning on {len(dataset)} images...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            
        print(f"Epoch {epoch+1}: Loss: {total_loss/len(loader):.4f}, Acc: {correct/len(dataset):.4f}")

    # Save fine-tuned
    torch.save(model.state_dict(), 'models/fine_tuned_model.pth')
    print("Fine-tuned model saved to models/fine_tuned_model.pth")

if __name__ == "__main__":
    fine_tune_classifier()
