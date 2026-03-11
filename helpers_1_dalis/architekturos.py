from torch import nn


class KetvirtaArchitektura(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3),  # (1, 28, 28) → (32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (32, 26, 26) → (32, 13, 13)
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3),  # (32, 13, 13) → (64, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (64, 11, 11) → (64, 5, 5)
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3), # (64, 5, 5) → (128, 3, 3)
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),          # (128, 3, 3) → 1152
            nn.Linear(1152, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class PenktaArchitektura(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3),  # (1, 28, 28) → (32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (32, 26, 26) → (32, 13, 13)
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3),  # (32, 13, 13) → (64, 11, 11)
            nn.ReLU(),
            nn.AvgPool2d(2),                   # (64, 11, 11) → (64, 5, 5)
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3), # (64, 5, 5) → (128, 3, 3)
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),          # (128, 3, 3) → 1152
            nn.Dropout(p=0.4),        # Add dropout for regularization
            nn.Linear(1152, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class AstuntaArchitektura(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3),  # (1, 28, 28) → (32, 26, 26)
            nn.ReLU(),
            nn.AvgPool2d(2),                   # (32, 26, 26) → (32, 13, 13)
            nn.BatchNorm2d(32),                # Add batch normalization

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3),  # (32, 13, 13) → (64, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (64, 11, 11) → (64, 5, 5)
            nn.BatchNorm2d(64),                # Add batch normalization
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),          # (128, 5, 5) → 1600
            nn.Linear(1600, 128),
            nn.BatchNorm1d(128),               # Add batch normalization
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x