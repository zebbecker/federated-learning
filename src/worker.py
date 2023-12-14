import sys
import time
import socket
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
import threading
import random

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchsummary import summary
from torch.optim import Adam

import numpy as np

import worker_model

PORT = 8083
BATCH_SIZE = 128
LEARNING_RATE = 0.001
# max number of times worker will try to reconnect to coordinator if an issue occurs during training.
MAX_RETRIES = 5

# Check if GPU is available, and use if possible, data is sent to "device" later
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

"""
Usage: python3 worker.py coordinator_ip:port worker_ip
"""


class SimpleWorkerServer(SimpleXMLRPCServer):
    def serve_forever(self):
        self.quit = False
        while not self.quit:
            self.handle_request()


class Worker:
    def __init__(self, ip_address, coordinator_hostname):
        # Set up RPC server to receive notifications
        self.ip_address = ip_address
        self.hostname = (
            "http://" + ip_address + ":" + str(PORT)
        )  # Public IP address that the coordinator should use to connect

        # Hackish, but gets the private IP address of the Amazon EC2 machines
        self.server = SimpleWorkerServer(
            (socket.gethostbyname(socket.gethostname()), PORT)
        )
        self.update_ready = False

        # Coordinator and Model Stuff
        # Filled in by connect with info from coordinator
        self.coordinator_hostname = coordinator_hostname
        self.coordinator = None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.epochs = 0
        self.global_epoch = 0
        self.testing = False

        # Set up data using PyTorch DataLoaders
        # Just a helpful object for splitting up data
        self.train_dl = None
        self.test_dl = None

        # Can always switch to local files later,
        # Gonna put this here for now (like every worker has the entire dataset)
        # Set download=True to download files if they are not on machines already
        mnist_train = MNIST(
            "~/data", train=True, download=True, transform=transforms.ToTensor()
        )
        mnist_test = MNIST(
            "~/data", train=False, download=True, transform=transforms.ToTensor()
        )

        # Get random sized subset
        set_size = random.randint(100, len(mnist_train))
        random_set = set(random.sample(range(1, len(mnist_train)), set_size))
        subset = Subset(mnist_train, random_set)
        print(f"Working with dataset of {set_size} images")

        self.train_dl = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        self.test_dl = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True)

    def connect(self):
        """Establish connection to coordinator"""
        print(
            "["
            + self.ip_address
            + "] Worker attemping to connect to "
            + self.coordinator_hostname
        )

        # Start up worker server in seperate thread
        self.server.register_function(self.receive_notification, "notify")
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.start()
        print("[" + self.ip_address + "] Started worker server in seperate thread")

        # Connect to host and greet with intro message
        self.coordinator = xmlrpc.client.ServerProxy(self.coordinator_hostname)

        retries = MAX_RETRIES
        while retries > 0:
            try:
                self.testing = self.coordinator.connect(
                    self.hostname, len(self.train_dl)
                )
                print(
                    "[" + self.ip_address + "] Connected to", self.coordinator_hostname
                )
                break
            except Exception as e:
                print(
                    "[" + self.ip_address + "] Error: Unable to connect to",
                    self.coordinator_hostname,
                )
                retries -= 1
                if retries < 0:
                    raise e
                time.sleep(1)

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
        print("[" + self.ip_address + "] Training ...")
        loss_history = []
        start = time.time()
        for epoch in range(self.epochs):
            print(f"[{self.ip_address}] Running Epoch {epoch + 1} of {self.epochs}")
            epoch_losses = []
            for batch in self.train_dl:
                # Check if coordinator has sent an update
                if self.update_ready:
                    return None

                # Train a batch
                # x, y = batch
                # x, y = x.to(device), y.to(device)
            #     batch_loss = self.train_batch(x, y)
            #     epoch_losses.append(batch_loss)

            # epoch_loss = np.mean(epoch_losses)
            # loss_history.append(epoch_loss)

        end = time.time()
        training_time = end - start
        print(
            f"[{self.ip_address}] Loss History: {loss_history}"
        )  # Could be useful if we want to plot loss later
        print(
            f"[{self.ip_address}] Training Time: {training_time}"
        )  # Could be useful for tests later

        return [param.data.tolist() for param in self.model.parameters()]

    def test(self):
        """Test the model to get accuracy"""

        # Switch flag off
        self.accuracy_requested = False

        print("[" + self.ip_address + "] Testing Model...")

        # Set the model to evaluation mode
        self.model.eval()

        correct_predictions = 0
        total_samples = 0

        # with torch.no_grad():
        #     for inputs, labels in self.test_dl:
        #         # Forward pass
        #         inputs = inputs.to(device)
        #         outputs = self.model(inputs)

        #         # Get predictions
        #         _, predicted = torch.max(outputs, 1)

        #         # Update counts
        #         total_samples += labels.size(0)
        #         correct_predictions += (predicted == labels.to(device)).sum().item()

        # Calculate accuracy
        if total_samples > 0:
            # Avoid divide by zero error when debugging
            return correct_predictions / total_samples
        else:
            return 0

    def receive_notification(self, notification):
        if notification == "Update Ready":
            self.update_ready = True
            return "Update Received"

        elif notification == "Ping":
            return "Pong"

        elif notification == "Shutdown":
            self.server.quit = True
            return "Shutting down"

    def wait_for_notification(self):
        # Wait for server thread to register an update
        while not self.server.quit and not self.update_ready:
            time.sleep(1)
            if self.coordinator.update_ready(self.hostname, self.global_epoch) == "Yes":
                self.update_ready = True

        # Reset update status for later
        self.update_ready = False

    def work(self):
        retries = MAX_RETRIES
        """Main loop for working with coordinator"""
        while not self.server.quit:
            # Get caught up to date with coordinator
            try:
                update, epoch, num_epochs = self.coordinator.get_update(self.hostname)
                self.update_weights(update)
                self.global_epoch = epoch
                self.epochs = num_epochs
            except Exception as e:
                print(f"[{self.ip_address}] Problem while updating: {e}")
                if retries < 0:
                    break
                else:
                    print("Retrying...")
                    retries -= 1
                    time.sleep(1)
                    continue

            # Train on local data and push contribution
            try:
                print(
                    f"[{self.ip_address}] Training for Global Epoch: {self.global_epoch}"
                )
                new_weights = self.train()
                if not new_weights:
                    print(f"[{self.ip_address}] Training interrupted by update")
                else:
                    accuracy = self.test() if self.testing else None
                    status = self.coordinator.load_update(
                        self.hostname, new_weights, self.global_epoch, accuracy
                    )
                    if status == "Error: Worker not registered":
                        self.connect()
                        continue
                    if status != "Ok":
                        print(
                            f"[{self.ip_address}] Coordinator could not use update: {status}"
                        )
                        # @TODO dont just shutdown: try to connect again and keep working
                        # @TODO maybe this is what is stalling?
                        # break
                        # self.wait_for_notification()
                        # continue
                        # Note that this flow never calls wait for notification. not sure if that is an issue.
                        # Wait for notification just blocks this loop from restarting until an update is ready.
                self.wait_for_notification()
            except Exception as e:
                print(f"[{self.ip_address}] Problem while training: {e}")
                if retries < 0:
                    break
                else:
                    print("Retrying...")
                    retries -= 1
                    time.sleep(1)
                    continue

        # Clean up worker server if exited with training issue
        if not self.server.quit:
            self.coordinator.disconnect(self.hostname)
            self.server.quit = True


def main():
    if len(sys.argv) != 3:
        print("Usage: python worker.py coordinator_ip:port worker_ip")
        sys.exit(1)

    name = sys.argv[1]
    coordinator_hostname = "http://" + name
    worker_ip = sys.argv[2]
    worker = Worker(worker_ip, coordinator_hostname)

    print(f"[{worker_ip}] Running on {device}")

    try:
        worker.connect()
    except ConnectionRefusedError as e:
        print(f"[{worker_ip}] Couldn't Connect: {e}")

    worker.work()
    print("[" + worker_ip + "] Training complete. Shutting down.")


if __name__ == "__main__":
    main()
