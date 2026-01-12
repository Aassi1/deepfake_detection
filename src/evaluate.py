import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from tqdm import tqdm

from model import get_model
from create_splits import create_splits
from split_dataset import create_datasets_from_splits
from transforms import get_val_test_transforms


def evaluate_model(model, dataloader, device):
    """
    Evaluate model and return predictions + labels
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_probs, save_path='results/roc_curve.png'):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()
    
    return roc_auc


def print_metrics(y_true, y_pred, cm):
    """Print detailed metrics"""
    # Overall accuracy
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    
    # Real class metrics (class 0)
    precision_real = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0
    
    # Fake class metrics (class 1)
    precision_fake = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_fake = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (Real predicted as Real):  {tn:,}")
    print(f"  False Positives (Real predicted as Fake): {fp:,}")
    print(f"  False Negatives (Fake predicted as Real): {fn:,}")
    print(f"  True Positives (Fake predicted as Fake):  {tp:,}")
    
    print(f"\nPer-Class Metrics:")
    print(f"\n  REAL Class (0):")
    print(f"    Precision: {precision_real*100:.2f}%")
    print(f"    Recall:    {recall_real*100:.2f}%")
    print(f"    F1-Score:  {f1_real*100:.2f}%")
    
    print(f"\n  FAKE Class (1):")
    print(f"    Precision: {precision_fake*100:.2f}%")
    print(f"    Recall:    {recall_fake*100:.2f}%")
    print(f"    F1-Score:  {f1_fake*100:.2f}%")
    
    print("\n" + "="*60)
    
    # Also use sklearn's classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Real', 'Fake'],
                                digits=4))


def main():
    # Create results folder
    import os
    os.makedirs('results', exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading test data...")
    splits = create_splits("data/faces/real", "data/faces/fake")
    val_transform = get_val_test_transforms()
    
    _, _, test_dataset = create_datasets_from_splits(
        splits, val_transform, val_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Load model
    print("\nLoading trained model...")
    model = get_model(device, pretrained=False, freeze_backbone=True)
    
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val acc: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)
    
    # Print metrics
    cm = plot_confusion_matrix(y_true, y_pred)
    print_metrics(y_true, y_pred, cm)
    
    # ROC curve
    roc_auc = plot_roc_curve(y_true, y_probs)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    print("\n Evaluation complete! Results saved to 'results/' folder")


if __name__ == "__main__":
    main()