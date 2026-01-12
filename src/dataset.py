import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DeepfakeDataset(Dataset):
    """Dataset for loading deepfake face images"""
    
    def __init__(self, real_folder, fake_folder, transform=None):
        """
        Args:
            real_folder: Path to folder with real face images
            fake_folder: Path to folder with fake face images
            transform: Optional transforms to apply to images
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        real_files = os.listdir(real_folder)
        for filename in real_files:
            if filename.endswith('.jpg'):
                full_path = os.path.join(real_folder, filename)
                self.image_paths.append(full_path)
                self.labels.append(0)  
        
        fake_files = os.listdir(fake_folder)
        for filename in fake_files:
            if filename.endswith('.jpg'):
                full_path = os.path.join(fake_folder, filename)
                self.image_paths.append(full_path)
                self.labels.append(1)  
        
        print(f"Loaded {len(self.image_paths)} images")
        print(f"  Real: {self.labels.count(0)}")
        print(f"  Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label