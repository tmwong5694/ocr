import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TransferResNet(nn.Module):

    def __init__(self, num_classes=2, freeze=False):
        super(TransferResNet, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get the number of input features for the final fully connected layer
        num_ftrs = self.model.fc.in_features
        # model.fc is the final fully_connected layer
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Remove the final layer to get embeddings
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        return self.model(x)

    # TODO: check what does it do later
    def get_embedding(self, x):
        return self.feature_extractor(x).squeeze()


if __name__ == "__main__":
    import torch

    # Create the model
    model = TransferResNet(num_classes=2, freeze=True)
    print(f"Model created. Final layer: {model.model.fc}")

    # Test with a dummy tensor (Batch Size=1, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [1, 2]

    embedding = model.get_embedding(dummy_input)
    print(f"Embedding shape: {embedding}")