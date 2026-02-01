import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics

def calculate_per_class_metrics(y_true, y_pred, class_names):
    """
    Calculate per-class metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary with per-class metrics
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    
    return per_class

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return cm

def plot_roc_curves(y_true, y_pred_proba, class_names, save_path=None):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        save_path: Path to save plot (optional)
    """
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels for multi-class ROC
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Dictionary with train/val loss and accuracy
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("="*60)

def calculate_class_distribution(labels, class_names):
    """
    Calculate and display class distribution
    
    Args:
        labels: Array of labels
        class_names: List of class names
    
    Returns:
        Dictionary with class counts
    """
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    for i, name in enumerate(class_names):
        count = distribution.get(i, 0)
        percentage = (count / len(labels)) * 100
        print(f"{name:25s}: {count:5d} ({percentage:5.2f}%)")
    print("="*60)
    print(f"{'Total':25s}: {len(labels):5d} (100.00%)")
    print("="*60)
    
    return distribution

def calculate_model_size(model):
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

if __name__ == "__main__":
    # Test metrics calculation
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 0, 1, 3, 3])
    class_names = ['Class A', 'Class B', 'Class C', 'Class D']
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("Overall Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Per-class metrics
    per_class = calculate_per_class_metrics(y_true, y_pred, class_names)
    print("\nPer-class Metrics:")
    for name, values in per_class.items():
        print(f"{name}:")
        for metric, score in values.items():
            print(f"  {metric}: {score:.4f}")
    
    # Classification report
    print_classification_report(y_true, y_pred, class_names)