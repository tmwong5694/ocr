import torch
from torch import Tensor
import torchvision.models as models
from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from src.DeepLearning.time_utils import timeit

# import torch._dynamo
# # Suppress errors during compilation
# torch._dynamo.config.suppress_errors = True

# Set "mps" as the accelerator
device = torch.device("mps")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

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
# Optimal batch size for stable gradient descent
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Get the length of dataset for calculation of accuracy
train_N, test_N = len(train_dataset), len(test_dataset)

# Define crtierion of cross entropy loss
criterion = nn.CrossEntropyLoss()
# Use Adam as optimizer and learning rate of 0.001
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

def get_batch_accuracy(outputs: Tensor, labels: Tensor, total_num: int) -> float:
    """Take predictions and labels in batch to calculate accuracy per batch"""
    # No. of columns refer to the no. of classes
    # The predicted class will have the highest probability across columns
    pred = outputs.argmax(dim=1, keepdim=True)
    # Pairwise comparison of predictions and labels, return true if prediction is correct
    correct = pred.eq(labels.view_as(pred)).sum().item()
    return correct / total_num

@timeit
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        running_loss = 0
        accuracy = 0
        for batch, (inputs, labels) in enumerate(train_loader):
            print(f"Batch: {batch + 1}/{len(train_loader)}")
            inputs, labels = inputs.to(device), labels.to(device)

            #
            optimizer.zero_grad()
            # Predict output class using input
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, labels)
            # Back propagation
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy += get_batch_accuracy(outputs=outputs, labels=labels, total_num=train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))
    return model

if __name__ == "__main__":

    model.to(device)
    # model = torch.compile(model)
    train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)



    pass