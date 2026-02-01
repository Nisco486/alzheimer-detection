
"""
CNN-Only Model for Alzheimer's Detection
Architecture: EfficientNet-B4 with custom classification head
Target: 90%+ accuracy on OASIS dataset

Save this as: models/cnn_backbone.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

class CNNModel(nn.Module):
    """
    CNN-only model using EfficientNet-B4 backbone
    For comparison with Hybrid and ViT models
    """
    def __init__(self, num_classes=4, pretrained=True, dropout=0.3):
        super(CNNModel, self).__init__()
        
        # Load pre-trained EfficientNet-B4
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # Remove global pooling
        )
        
        # Get number of features from backbone
        # EfficientNet-B4 has 1792 features
        self.feature_dim = 1792
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            logits: Class logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)  # [B, 1792, H', W']
        
        # Global pooling
        features = self.global_pool(features)  # [B, 1792, 1, 1]
        features = features.flatten(1)  # [B, 1792]
        
        # Classification
        logits = self.classifier(features)  # [B, num_classes]
        
        return logits
    
    def get_features(self, x):
        """
        Extract features without classification (for visualization)
        """
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.flatten(1)
        return features


# ==================== TRAINING SCRIPT ====================
if __name__ == "__main__":
    """
    Complete training script for CNN-only model
    Run this to train and save weights
    """
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings('ignore')
    
    # ==================== CONFIGURATION ====================
    class Config:
        # Paths - UPDATED TO USER'S ACTUAL PATH
        DATA_DIR = 'data/raw' 
        CHECKPOINT_DIR = 'checkpoints/cnn_only'
        
        # Dataset
        IMG_SIZE = 224
        NUM_CLASSES = 4
        CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        
        # Training
        BATCH_SIZE = 32
        NUM_EPOCHS = 50
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 0.01
        
        # Model
        DROPOUT = 0.3
        
        # Device
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        SEED = 42
    
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    
    # ==================== DATASET CLASS ====================
    class OASISDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            try:
                img_path = self.image_paths[idx]
                image = Image.open(img_path).convert('RGB')
                label = self.labels[idx]
                
                if self.transform:
                    image = self.transform(image)
                    
                return image, label
            except Exception as e:
                # Return dummy tensor on error to prevent crash
                return torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE), 0
    
    # ==================== DATA TRANSFORMS ====================
    def get_transforms(train=True):
        if train:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(Config.IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    # ==================== TRAINING FUNCTIONS ====================
    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return running_loss / len(dataloader), 100. * correct / total
    
    def validate(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation', leave=False):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return running_loss / len(dataloader), 100. * correct / total, all_preds, all_labels
    
    # ==================== DATA LOADING ====================
    def load_dataset():
        """Load dataset paths and labels"""
        image_paths = []
        labels = []
        label_map = {name: idx for idx, name in enumerate(Config.CLASS_NAMES)}
        
        print("Loading dataset...")
        # Walk through the directory structure
        # Updated to handle standard structure: root/class/*.jpg
        if not os.path.exists(Config.DATA_DIR):
            print(f"Directory not found: {Config.DATA_DIR}")
            return np.array([]), np.array([])
            
        for class_name in Config.CLASS_NAMES:
            class_dir = os.path.join(Config.DATA_DIR, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
                
            # Find all images
            class_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                import glob
                class_images.extend(glob.glob(os.path.join(class_dir, ext)))
                
            print(f"{class_name}: {len(class_images)} images")
            
            for img_path in class_images:
                image_paths.append(str(img_path))
                labels.append(label_map[class_name])
        
        return np.array(image_paths), np.array(labels)
    
    # ==================== MAIN TRAINING ====================
    print("=" * 70)
    print(" CNN-ONLY MODEL TRAINING (EfficientNet-B4)")
    print("=" * 70)
    print(f"\\nDevice: {Config.DEVICE}")
    
    # Load data
    image_paths, labels = load_dataset()
    print(f"Found {len(image_paths)} images.")
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {Config.DATA_DIR}. Please check the path.")
    else:
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=0.15, stratify=labels, random_state=Config.SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=Config.SEED
        )
        
        print(f"\\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Calculate class weights
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) > 0:
            class_weights = len(y_train) / (len(unique) * counts)
            class_weights = class_weights / class_weights.min()
            class_weights = torch.FloatTensor(class_weights).to(Config.DEVICE)
            print("\\nClass Weights:", class_weights.cpu().numpy())
        else:
            class_weights = None
            print("Warning: Could not calculate class weights.")
        
        # Create datasets
        train_dataset = OASISDataset(X_train, y_train, get_transforms(train=True))
        val_dataset = OASISDataset(X_val, y_val, get_transforms(train=False))
        test_dataset = OASISDataset(X_test, y_test, get_transforms(train=False))
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0) # reduced workers for windows
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Initialize model
        print("\\nInitializing CNN Model (EfficientNet-B4)...")
        model = CNNModel(num_classes=Config.NUM_CLASSES, pretrained=True, dropout=Config.DROPOUT).to(Config.DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
        
        # Training loop
        print("\\n" + "=" * 70)
        print("Starting Training...")
        print("=" * 70)
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            print("-" * 70)
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, Config.DEVICE)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model weights
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'model_config': {
                        'num_classes': Config.NUM_CLASSES,
                        'dropout': Config.DROPOUT
                    }
                }, f'{Config.CHECKPOINT_DIR}/cnn_best.pth')
                print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\\nEarly stopping at epoch {epoch+1}")
                break
        
        # Test evaluation
        print("\\n" + "=" * 70)
        print("Testing...")
        print("=" * 70)
        
        try:
            checkpoint = torch.load(f'{Config.CHECKPOINT_DIR}/cnn_best.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, Config.DEVICE)
            
            print(f"\\nTest Accuracy: {test_acc:.2f}%")
            print("\\nClassification Report:")
            print(classification_report(test_labels, test_preds, target_names=Config.CLASS_NAMES, digits=4))
            
            print("\\n" + "=" * 70)
            print(f"✓ Training Complete!")
            print(f"✓ Best Model Saved: {Config.CHECKPOINT_DIR}/cnn_best.pth")
            print(f"✓ Final Test Accuracy: {test_acc:.2f}%")
            print("=" * 70)
        except FileNotFoundError:
            print("Error: Best model checkpoint not found. Training might have failed.")
