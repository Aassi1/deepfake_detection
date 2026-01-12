import os
import random
from dataset import DeepfakeDataset


"""
This file will split our data into training, testing and validation sections in order to train the CNN

"""
def create_splits(real_folder, fake_folder, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create train/val/test splits from image folders
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all real image paths
    real_files = [f for f in os.listdir(real_folder) if f.endswith('.jpg')]
    real_paths = [os.path.join(real_folder, f) for f in real_files]
    
    # Get all fake image paths
    fake_files = [f for f in os.listdir(fake_folder) if f.endswith('.jpg')]
    fake_paths = [os.path.join(fake_folder, f) for f in fake_files]
    
    # Shuffle
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    # Calculate split indices for REAL images
    n_real = len(real_paths)
    train_end_real = int(n_real * train_ratio)
    val_end_real = int(n_real * (train_ratio + val_ratio))
    
    real_train = real_paths[:train_end_real]
    real_val = real_paths[train_end_real:val_end_real]
    real_test = real_paths[val_end_real:]
    
    # Calculate split indices for FAKE images
    n_fake = len(fake_paths)
    train_end_fake = int(n_fake * train_ratio)
    val_end_fake = int(n_fake * (train_ratio + val_ratio))
    
    fake_train = fake_paths[:train_end_fake]
    fake_val = fake_paths[train_end_fake:val_end_fake]
    fake_test = fake_paths[val_end_fake:]
    
    # Combine and create splits info
    splits = {
        'train': {'real': real_train, 'fake': fake_train},
        'val': {'real': real_val, 'fake': fake_val},
        'test': {'real': real_test, 'fake': fake_test}
    }
    
    # Print statistics
    print("="*50)
    print("DATA SPLITS CREATED")
    print("="*50)
    print(f"Train: {len(real_train)} real + {len(fake_train)} fake = {len(real_train) + len(fake_train)} total")
    print(f"Val:   {len(real_val)} real + {len(fake_val)} fake = {len(real_val) + len(fake_val)} total")
    print(f"Test:  {len(real_test)} real + {len(fake_test)} fake = {len(real_test) + len(fake_test)} total")
    print(f"\nTotal: {n_real + n_fake} images")
    print("="*50)
    
    return splits

if __name__ == "__main__":
    splits = create_splits("data/faces/real", "data/faces/fake")