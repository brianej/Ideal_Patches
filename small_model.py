import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallModel(nn.Module):
    def __init__(self, input_shape, patch_sizes):
        super().__init__()
        
        self.c, self.h, self.w = input_shape
        self.num_patches = len(patch_sizes)
        self.expanded = self.c * self.num_patches
        
        # Encoder layers with convolutional and pooling operations
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # [B, 32, H/2, W/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # [B, 64, H/4, W/4]
        ) 
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),  # [B, 64, H/4, W/4]
            nn.ReLU(inplace=True),
        )

        # 1x1x1 Convolutional layer
        self.score_layer = nn.Conv2d(64, self.expanded, kernel_size=1) # [B, C*S, H/4, W/4]
        
        # Upsample
        self.up2 = nn.ConvTranspose2d(self.expanded, self.expanded, 2, stride=2)  # [B, S*C, H/2, W/2]
        self.up1 = nn.ConvTranspose2d(self.expanded, self.expanded, 2, stride=2)  #[B, S*C, H, W]
        
        self.softmax = nn.Softmax(dim=1)  # Softmax to normalise scores across patches

    def forward(self, x):
        # Convolutional and pooling layers
        b, _, _, _ = x.shape
        x = self.layer1(x)

        x = self.bottleneck(x)

        # 1x1x1 Convolutional layer to get scores for each patch size
        x = self.score_layer(x)  # [B, S*C, H/4, W/4]
        x = F.relu(x, inplace=True)
        
        # Upsample 
        x = self.up2(x)     # [B, S*C, H/2, W/2]
        x = self.up1(x)     # [B, S*C, H, W]

        # Reorder it
        x = x.view(b, self.num_patches, self.c, self.h, self.w) # [B, S, C, H, W]
        
        # Apply softmax to normalise scores across patches
        x = self.softmax(x)
        
        return x
    