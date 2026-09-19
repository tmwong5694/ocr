import torch.nn as nn


class MNISTClassifier(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 64 channels * 7 height * 7 width = 3136
            nn.Linear(3136, num_classes)  # 10 output classes for digits 0-9
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
