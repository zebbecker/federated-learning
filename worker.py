"""Script for Worker Machines to Run

Idea is for workers to connect to coordinator then iteratively train
1. Send introduction to coordinator
2. Initialize model from coordinator's spec
3. Get global update from coordinator
4. Train on local data
5. Send update to coordinator 
6. Go back to step 3 and loop for lifetime of worker

TODO
- Right now Worker assumes that coordinator has connect and update methods
- Need to figure out data formatting, to make sure things are marshalled correctly
- Eventually change data to work with local files
- Eventually add a way to dynamically create model architecture
"""

import sys
import time
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
import threading

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from torch.optim import Adam

import numpy as np

import worker_model

PORT = 8082
BATCH_SIZE = 128
LEARNING_RATE = 0.001

EPOCHS_PER_UPDATE = 1

# Check if GPU is available, and use if possible, data is sent to "device" later
device = "cuda" if torch.cuda.is_available() else "cpu"


class Worker:
    def __init__(self, ip_address):
        # Set up RPC server to receive notifications
        self.hostname = (
            "http://" + ip_address + ":" + str(PORT)
        )  # address that worker server is serving on
        self.server = SimpleXMLRPCServer((ip_address, PORT))  # worker server
        self.update_ready = False

        # Coordinator and Model Stuff
        # Filled in by connect with info from coordinator
        self.coordinator = None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.epochs = EPOCHS_PER_UPDATE
        self.global_epoch = 0

        # Set up data using PyTorch DataLoaders
        # Just a helpful object for splitting up data
        self.train_dl = None
        self.test_dl = None

        # Can always switch to local files later,
        # Gonna put this here for now (like every worker has the entire dataset)
        mnist_train = MNIST(
            "~/data", train=True, download=True, transform=transforms.ToTensor()
        )
        mnist_test = MNIST(
            "~/data", train=False, download=True, transform=transforms.ToTensor()
        )
        self.train_dl = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
        self.test_dl = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True)

    def connect(self, coordinator_hostname):
        """Establish connection to coordinator"""

        print("Worker attemping to connect to " + coordinator_hostname)

        # Start up worker server in seperate thread
        self.server.register_function(self.receive_notification, "notify")
        self.server.register_function(self.ping, "ping")
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.start()
        print("Started worker server in seperate thread")

        # Connect to host and greet with intro message
        # self.coordinator = xmlrpc.client.ServerProxy(self.hostname)
        self.coordinator = xmlrpc.client.ServerProxy(coordinator_hostname)

        try:
            self.coordinator.connect(self.hostname)
            print("Connected to", coordinator_hostname)
        except Exception as e:
            print("Error: Unable to connect to", coordinator_hostname)
            raise e

        # Initialize model as it is defined in worker_model.py
        self.model = worker_model.model.to(device)
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.CrossEntropyLoss().to(device)

    def update_weights(self, coordinator_weights):
        """Update PyTorch Model from raw weights"""

        old_state = self.model.state_dict()
        new_state = old_state.copy()
        for name, new_weights in zip(old_state, coordinator_weights):
            old_tensor = new_state[name]
            new_tensor = torch.tensor(
                new_weights, dtype=old_tensor.dtype, device=device
            )
            new_state[name] = new_tensor

        self.model.load_state_dict(new_state)

    def train_batch(self, x, y):
        """Given data and labels for a batch, learn better weights"""

        self.optimizer.zero_grad()  # Flush memory
        pred = self.model(x)  # Get predictions
        batch_loss = self.loss(pred, y)  # Compute loss
        batch_loss.backward()  # Compute gradients
        self.optimizer.step()  # Make a GD step
        return batch_loss.detach().cpu().numpy()

    def train(self):
        """Run through local training data for a bit

        Runs for as many epochs as is specified in self.epochs. Each
        epoch is split up into batches for SGD and shuffling the data.
        Returns the new weights for the model as nested list
        """

        # print("Training with data", self.data)
        print("Training ...")
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
        print(
            f"Loss History: {loss_history}"
        )  # Could be useful if we want to plot loss later
        print(f"Training Time: {training_time}")  # Could be useful for tests later

        return [param.data.tolist() for param in self.model.parameters()]

    def receive_notification(self):
        self.update_ready = True

    def ping(self):
        return "pong"

    def wait_for_notification(self):
        # Wait for server thread to register an update
        while not self.update_ready:
            time.sleep(0.01)

        # Reset update status for later
        self.update_ready = False

    def work(self):
        """Main loop for working with coordinator"""

        while True:
            # Get caught up to date with coordinator
            try:
                update, epoch = self.coordinator.get_update()
                self.update_weights(update)
                self.global_epoch = epoch
            except Exception as e:
                print(f"Problem while updating: {e}")
                break

            # Train on local data and push contribution
            try:
                new_weights = self.train()
                status = self.coordinator.load_update(new_weights)
                if status != "Ok":
                    print(f"Coordinator could not use update: {status}")
                    break
                self.wait_for_notification()

            except Exception as e:
                print(f"Problem while training: {e}")
                break

        # Clean up worker server - allows thread to complete
        self.coordinator.disconnect(self.hostname)
        self.server.shutdown()
        self.server.server_close()


def main():
    print(f"Running on {device}")
    if len(sys.argv) != 3:
        print("Usage: python worker.py coordinator_ip:port worker_ip")
        sys.exit(1)

    # Get hostname from command line "http://<hostname>:<port>"
    name = sys.argv[1]
    coordinator_hostname = "http://" + name

    worker_ip = sys.argv[2]
    worker = Worker(worker_ip)

    try:
        worker.connect(coordinator_hostname)
    except ConnectionRefusedError as e:
        print(f"Couldn't Connect: {e}")

    worker.work()
    print("Finished working")


if __name__ == "__main__":
    main()
