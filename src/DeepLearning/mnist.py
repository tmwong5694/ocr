import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision import datasets
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

import torch._dynamo
# Suppress compilation error
torch._dynamo.config.suppress_errors = True

# Set "mps" as the accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # Output with a class of (batch_size, num_classes)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        # Backpropogation to calculate weight
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print("Train - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

trans = v2.Compose([
    # v2.ToTensor() is deprecated
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform = trans
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform = trans
)

x_0, y_0 = train_data[0]

x_0_tensor = trans(x_0)
x_0_gpu = x_0_tensor.to(device)
image = F.to_pil_image(x_0_gpu)
plt.imshow(image, cmap="gray")

# train_data.transforms = trans
# test_data.transforms = trans

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(test_data, batch_size=batch_size)


# test_matrix = torch.tensor(
#     [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# )
# print(nn.Flatten()(test_matrix[None, :]))
# print(nn.Flatten()(test_matrix))


input_size = 1 * 28 * 28
n_classes = 10
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, n_classes)
]

model = nn.Sequential(*layers)
model.to(device)

# Compilation error is suppressed
model = torch.compile(model)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())


train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)



if __name__ == '__main__':
    epochs = 10

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        train()
        validate()

    # Original: prediction = model(x_0_gpu)
    prediction = model(x_0_gpu)


    print(f"Prediction: {prediction.argmax(dim=1, keepdim=True)}")

    print(y_0)

pass