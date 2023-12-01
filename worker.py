"""Script for Worker Machines to Run

Idea is for workers to connect to coordinator then iteratively train
1. Send introduction to coordinator
2. Initialize model from coordinator's spec
3. Train on local data
4. Send update to coordinator and receive global update
5. Go back to step 3 and loop for lifetime of worker
"""

import sys
import time
import xmlrpc.client

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary
from torch.optim import Adam

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class Worker:

    def __init__(self):
        self.data = None
        self.train_dl = None
        self.test_dl = None

        self.server = None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.epochs = 1

    def connect(self, hostname):
        """Establish connection to coordinator"""

        # Connect to host and greet with intro message
        self.server = xmlrpc.client.ServerProxy(hostname)
        try:
            model_info = self.server.Coordinator.connect()
            print("Connected to", hostname)
        except ConnectionRefusedError as e:
            print("Error: Unable to connect to", hostname)
            raise e

        # Initialize model with info from coordinator
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), # in_ch, out_ch, k, stride, pad
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 10)
        ).to(device)  # Assuming the worker knows the model format...
        self.optimizer = Adam(self.model.parameters(), lr=.001)
        self.loss = nn.CrossEntropyLoss().to(device)


    def update_weights(self, coordinator_weights):
        """Update PyTorch Model from raw weights"""

        old_state = self.model.state_dict()
        new_state = old_state.copy()
        for name, new_weights in zip(old_state, coordinator_weights):
            old_tensor = new_state[name]
            new_tensor = torch.tensor(new_weights, dtype=old_tensor.dtype, device=device)
            new_state[name] = new_tensor

        self.model.load_state_dict(new_state)


    def train_batch(self, x, y):
        self.optimizer.zero_grad()       # Flush memory
        pred = self.model(x)             # Get predictions
        batch_loss = self.loss(pred, y)  # Compute loss
        batch_loss.backward()            # Compute gradients
        self.optimizer.step()            # Make a GD step
        return batch_loss.detach().cpu().numpy()

    def train(self):
        """Run through local training data for a bit
        
        Runs for as many epochs as is specified in self.epochs. Each 
        epoch is split up into batches for SGD and shuffling the data.
        Returns the new weights for the model as nested list
        """
        
        print("Training with data", self.data)
        loss_history = []
        start = time.time()
        for epoch in range(self.epochs):

            print(f"Running Epoch {epoch + 1} of {self.epochs}")
            epoch_losses = []
            for batch in self.train_dl:
                x, y = batch
                x, y = x.to(device), y.to(device)
                batch_loss = self.train_batch(x, y)
                epoch_losses.append(batch_loss)

            epoch_loss = np.mean(epoch_losses)
            loss_history.append(epoch_loss)

        end = time.time()
        training_time = end - start
        print(f"Loss History: {loss_history}")    # Could be useful if we want to plot loss later
        print(f"Training Time: {training_time}")  # Could be useful for tests later
        
        return [param.data.tolist() for param in self.model.parameters()]


    def work(self):
        """Main loop for working with coordinator"""

        while True:
            new_weights = self.train()
            try:
                update = self.server.Coordinator.update(new_weights)
                self.update_weights(update)
            except Exception as e:
                print(f"Problem while training: {e}")
                break


def main():

    print(f"Running on {device}")
    worker = Worker()
    name = sys.argv[1]
    hostname = "http://" + name

    try:
        worker.connect(hostname)
    except ConnectionRefusedError as e:
        print(f"Couldn't Connect: {e}")

    worker.work()
    print("Finished working")


if __name__ == "__main__":
    main()
