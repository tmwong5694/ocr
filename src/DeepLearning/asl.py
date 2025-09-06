import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))

device = torch.device("mps")

data_folder = os.getenv("DATA_FOLDER")

train_df = pd.read_csv(os.path.join(data_folder, "sign_mnist_train.csv"))
valid_df = pd.read_csv(os.path.join(data_folder, "sign_mnist_valid.csv"))

y_train = train_df.pop("label")
y_valid = valid_df.pop("label")

x_train = train_df.to_numpy(dtype=np.int32)
x_valid = valid_df.to_numpy(dtype=np.int32)

# list(map(lambda x: print(x.shape), (x_train, y_train, x_valid, y_valid)))

# #TODO: figure out plotting
# plt.figure(figsize=(40,40))
# num_images = 20
# for i, (x_train_data, y_train_data) in enumerate(zip(x_train, y_train)):
#     if i == 20:
#         break
#     row = x_train_data
#     label = y_train_data

#     image = row.reshape(28, 28)
#     plt.subplot(1, num_images, i+1)
#     plt.title(label, fontdict={'fontsize': 30})
#     plt.axis('off')
#     plt.imshow(image, cmap='gray')


class MyDataset(Dataset):
    def __init__(self, x_df, y_df):
        """Load the pd dataframes into tensor using 'mps' device"""
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)
    
    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y
    
    def __len__(self):
        return len(self.xs)

BATCH_SIZE = 32
train_data = MyDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_N = len(train_loader.dataset)


valid_data = MyDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
valid_N = len(valid_loader.dataset)

batch = next(iter(train_loader))

input_size = 28 * 28
n_classes = 24

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, n_classes)
]
model = nn.Sequential(*layers)
model = torch.compile(model.to(device))
print(f"The model's parameters include: {model.parameters()}")

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

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
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()
        
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

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

import torch._dynamo
torch._dynamo.config.suppress_errors = True

epochs = 20

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()
