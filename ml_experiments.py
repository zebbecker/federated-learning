import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary
from torch.optim import Adam

import numpy as np
import time

# Check if GPU is available, and use if possible, data is sent to "device" later
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load MNIST data
mnist_train = MNIST('~/data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('~/data', train=False, download=True, transform=transforms.ToTensor())
train_dl = DataLoader(mnist_train, batch_size=128, shuffle=True)

# Simple Model: 2 convolutional layers, flatten, then fully connected classifier
model = nn.Sequential(
    nn.Conv2d(1, 64, 3, 1, 1), # in_ch, out_ch, k, stride, pad
    nn.ReLU(),
    nn.Conv2d(64, 4, 3, 1, 1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(3136, 10)
).to(device)

summary(model, (1, 28, 28))


# Train the simple model:
def train_batch(model, X_train, Y_train, opt, loss):
    opt.zero_grad()                   # Flush memory
    pred = model(X_train)             # Get predictions
    batch_loss = loss(pred, Y_train)  # Compute loss
    batch_loss.backward()             # Compute gradients
    opt.step()                        # Make a GD step
    return batch_loss.detach().cpu().numpy()


def train(model, train_dl, epochs, optimizer, loss):

    loss_history = []
    start = time.time()
    for epoch in range(epochs):

        print(f"Running Epoch {epoch + 1} of {epochs}")
        epoch_losses = []
        for batch in train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)
            batch_loss = train_batch(model, x, y, optimizer, loss)
            epoch_losses.append(batch_loss)

        epoch_loss = np.mean(epoch_losses)
        loss_history.append(epoch_loss)

    end = time.time()
    training_time = end - start
    return loss_history, training_time

loss = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters(), lr=.001)
loss_history, train_time = train(model, train_dl, 1, optimizer, loss)
print(loss_history)
print(train_time)


# Poke around and see what we can extract
params = model.parameters()
raw_weights = [param.data.tolist() for param in params]
# print(raw_weights[0])
# print(len(raw_weights))
# print(model.state_dict())


# Try to reconstruct model from weights
worker_copy = nn.Sequential(
    nn.Conv2d(1, 64, 3, 1, 1), # in_ch, out_ch, k, stride, pad
    nn.ReLU(),
    nn.Conv2d(64, 4, 3, 1, 1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(3136, 10)
).to(device)  # Assuming the worker knows the model format...

# Try to update weights using the raw weights
old_state = worker_copy.state_dict()
new_state = old_state.copy()
for name, new_weights in zip(old_state, raw_weights):
    old_tensor = new_state[name]
    new_tensor = torch.tensor(new_weights, dtype=old_tensor.dtype, device=device)
    new_state[name] = new_tensor

worker_copy.load_state_dict(new_state)


# Test Model
def accuracy(model, dataset):

    # Set the model to evaluation mode
    model.eval()

    correct_predictions = 0
    total_samples = 0

    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=True)
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Update counts
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels.to(device)).sum().item()

    # Calculate accuracy
    return correct_predictions / total_samples
print(f"Copy Accuracy: {accuracy(worker_copy, mnist_test)}")
print(f"Model Accuracy: {accuracy(model, mnist_test)}")