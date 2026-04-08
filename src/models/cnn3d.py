import torch
import torch.nn as nn

class SleepDisorderCNN(nn.Module):
    """
    3D Convolutional Neural Network designed to process EEG topomap sequences.
    
    Input shape: (Batch, 3, 30, 32, 32)
        - 3 Channels (Spectral Bands)
        - 30 Time Slices
        - 32x32 Spatial Grid
        
    The network extracts local spatiotemporal primitives in early layers and 
    integrates them into disorder-specific patterns in deeper layers.
    """

    def __init__(self, n_classes):
        super(SleepDisorderCNN, self).__init__()

        # --- Block 1: Local Pattern Extraction ---
        # Conv3D slides kernels across Time, Height, and Width.
        self.block1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: (32, 15, 16, 16)
        )

        # --- Block 2: Intermediate Feature Composition ---
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: (64, 7, 8, 8)
        )

        # --- Block 3: Deep Spatiotemporal Integration ---
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: (128, 3, 4, 4)
        )

        # --- Block 4: Global Representation ---
        self.block4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2), # Spatial dropout to handle noise in EEG channels
            nn.AdaptiveAvgPool3d(1), # Collapses into a single 256-dim feature vector
        )

        # --- Classifier: Decision Layer ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
            # Softmax is not included here as CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, 30, 32, 32).

        Returns:
            torch.Tensor: Logits for the n_classes.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
