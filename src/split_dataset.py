import torch
from torch.utils.data import Dataset
from PIL import Image

class SplitDeepfakeDataset(Dataset):
    """Dataset that works with pre-split image lists"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0=real, 1=fake)
            transform: Transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        assert len(image_paths) == len(labels), "Paths and labels must have same length"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_datasets_from_splits(splits, train_transform, val_transform):
    """
    Create train/val/test datasets from splits dictionary
    
    Args:
        splits: Dictionary from create_splits()
        train_transform: Transforms for training (with augmentation)
        val_transform: Transforms for val/test (no augmentation)
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Combine real and fake paths for each split
    train_paths = splits['train']['real'] + splits['train']['fake']
    train_labels = [0] * len(splits['train']['real']) + [1] * len(splits['train']['fake'])
    
    val_paths = splits['val']['real'] + splits['val']['fake']
    val_labels = [0] * len(splits['val']['real']) + [1] * len(splits['val']['fake'])
    
    test_paths = splits['test']['real'] + splits['test']['fake']
    test_labels = [0] * len(splits['test']['real']) + [1] * len(splits['test']['fake'])
    
    # Create datasets
    train_dataset = SplitDeepfakeDataset(train_paths, train_labels, train_transform)
    val_dataset = SplitDeepfakeDataset(val_paths, val_labels, val_transform)
    test_dataset = SplitDeepfakeDataset(test_paths, test_labels, val_transform)
    
    print(f"\nDatasets created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset