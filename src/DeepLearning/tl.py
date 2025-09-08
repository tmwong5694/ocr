import torch
import torchvision.models as models
from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import torch._dynamo

# Set "mps" as the accelerator
device = torch.device("mps")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
# Suppress errors during compilation
torch._dynamo.config.suppress_errors = True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # Predict output class using input
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model

if __name__ == "__main__":

    model.to(device)
    # model = torch.compile(model)
    train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)



    pass