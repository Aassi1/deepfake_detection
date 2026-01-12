import torch
import torch.nn as nn
from torchvision import models # This is where EfficientNet comes from 


class DeepfakeDetector(nn.Module):
    """EfficientNet-based deepfake detector"""
    
    def __init__(self, pretrained=True, freeze_backbone=True):

        # Initialize the parent nn.Module class
        super(DeepfakeDetector, self).__init__()
        
        # This is where the EfficientNet-B0 CNN with 5.3 milion params is loaded
        # These weights already know how to detect edges, texture and shapes
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # EfficientNet-B0s backbone ouptuts 1280 features 
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier head
        # Original EfficientNet: 1280 features -> 1000 classes (ImageNet)
        # Our custom head: 1280 features -> 2 classes (Real/Fake)
        self.backbone.classifier = nn.Sequential(
            # Dropout layer 1: Randomly turns off 30% of neurons during training
            # Purpose: Prevents overfitting by forcing model not to rely on specific neurons
            nn.Dropout(p=0.3),

            # Dense layer 1: Combines 1280 features into 512 intermediate values
            # This learns which combinations of features are important for detection
            nn.Linear(num_features, 512),


            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2)  # 2 classes: real, fake
        )
        
        # Optionally freeze backbone (only train new head)
        # "Freezing" means setting requires_grad=False
        # This prevents those weights from being updated during training
        if freeze_backbone:
            # loop through all the params in the feature extension layesr
            for param in self.backbone.features.parameters():

                #
                param.requires_grad = False
            print("Backbone frozen - only training classification head")
        else:
            print("Training entire network")
    
    def forward(self, x):
        return self.backbone(x)


def get_model(device, pretrained=True, freeze_backbone=True):
    """
    Create and return the model
    
    Args:
        device: torch.device (cuda or cpu)
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: If True, only train classification head
    
    Returns:
        model on specified device
    """

    # Creating the model isntance
    model = DeepfakeDetector(pretrained=pretrained, freeze_backbone=freeze_backbone)

    # Move model params to gpu since it is faster when training
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    return model