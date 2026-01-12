from torchvision import transforms

def get_train_transforms():
    """Transforms for training data (with augmentation)"""
    return transforms.Compose([
        transforms.RandomRotation(15),  # Rotate ±15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary brightness/contrast
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]     # ImageNet std
        )
    ])

def get_val_test_transforms():
    """Transforms for validation/test data (NO augmentation)"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])