import torch
import torch.nn as nn  # Neural network modules (loss functions, layers)
import torch.optim as optim  # Optimization algorithms (Adam, SGD, etc.)
from torch.utils.data import DataLoader  # Handles batching and loading data
from tqdm import tqdm  # Progress bars
import time  # For timing epochs

# Import our custom modules
from model import get_model  # Function to create our CNN
from create_splits import create_splits  # Splits data into train/val/test
from split_dataset import create_datasets_from_splits  # Creates PyTorch datasets
from transforms import get_train_transforms, get_val_test_transforms  # Data augmentation


def calculate_class_weights(train_dataset):
    """
    Calculate class weights for imbalanced dataset
    
    Problem: We have 10x more fake faces than real faces
    Solution: Give real faces higher weight during training
    
    Formula: weight_class = total_samples / (num_classes * samples_in_class)
    """
    labels = train_dataset.labels
    n_real = labels.count(0)
    n_fake = labels.count(1)
    total = len(labels)
    
    # Weight inversely proportional to class frequency
    weight_real = total / (2 * n_real)
    weight_fake = total / (2 * n_fake)

    # dtype=float32 because loss function expects float weights
    weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32)
    
    # Some statistics 
    print(f"\nClass distribution:")
    print(f"  Real: {n_real} ({n_real/total*100:.1f}%)")
    print(f"  Fake: {n_fake} ({n_fake/total*100:.1f}%)")
    print(f"\nClass weights:")
    print(f"  Real: {weight_real:.3f}")
    print(f"  Fake: {weight_fake:.3f}")
    
    return weights


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()  # Set to training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc='Training')
    
    for images, labels in pbar:
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()  # Set to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
     # torch.no_grad() context manager:
    #   - Don't compute gradients (saves memory and speeds up)
    #   - Don't track operations for backpropagation
    # We're only evaluating, not training, so we don't need gradients

    with torch.no_grad():  # No gradient calculation
        pbar = tqdm(dataloader, desc='Validation')
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    # Calculate validation metrics
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def main():
    # Configuration
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.0001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create splits
    print("Creating data splits...")
    splits = create_splits("data/faces/real", "data/faces/fake")
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_test_transforms()
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets_from_splits(
        splits, train_transform, val_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = get_model(device, pretrained=True, freeze_backbone=True)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'models/best_model.pth')
            print(f"  New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()