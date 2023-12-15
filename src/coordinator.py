from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from xmlrpc.client import ServerProxy

import numpy as np
import torch
import time
import socket

import worker_model

# Usage: python3 coordinator.py
# (set QP, coordinator IP, and port number in script below)

QUORUM_PERCENTAGE = 0.8

COORDINATOR_IP = "15.156.205.154"  # Public IP that workers should connect to
PORT = 8082

WORKER_STARTING_EPOCHS = 4

MAX_EPOCHS = 4  # Stop training after this many global epochs


def is_equal_dimensions(l1, l2):
    """Helper to compare two lists of tensors"""

    # Check number of tensors
    if len(l1) != len(l2):
        return False

    # Check each tensor size
    for t1, t2 in zip(l1, l2):
        if t1.size() != t2.size():
            return False

    return True


class WorkerInfo:
    """Class to hold info about each worker"""

    def __init__(self, hostname, data_size, num_epochs=WORKER_STARTING_EPOCHS):
        self.server = ServerProxy(hostname)
        self.num_epochs = num_epochs
        self.last_push = -1
        self.last_pull = -1
        self.hostname = hostname
        self.data_size = data_size
        self.epoch_start_time = 0
        self.epoch_end_time = 0
        self.epoch_durations = []

    def notify(self, notification):
        """Notify worker with some message"""
        return self.server.notify(notification)


class SimpleCoordinatorServer(SimpleXMLRPCServer):
    def serve_forever(self):
        self.quit = False
        while not self.quit:
            self.handle_request()


class Coordinator:
    def __init__(
        self, quorum_percentage=QUORUM_PERCENTAGE, max_epochs=MAX_EPOCHS, testing=False
    ):
        # Initialize weights using worker_model spec, list of tensors
        model = worker_model.model
        self.weights = [param.data for param in model.parameters()]

        # Set up RPC server
        self.server = SimpleCoordinatorServer(
            (socket.gethostbyname(socket.gethostname()), PORT),
            requestHandler=SimpleXMLRPCRequestHandler,
            logRequests=False,
        )

        # State for workers, updates, and training
        self.testing = testing
        self.workers = {}
        self.updates = {}
        self.epoch_accuracies = {}
        self.quorum_pct = quorum_percentage
        self.epoch = 1
        self.max_epochs = max_epochs

        # Record accuracy score on test task after each global epoch. Stored as (epoch, score) tuples
        self.accuracies = []
        self.epoch_start_time = time.time()
        self.epoch_end_time = time.time()

    def accept_connection(self, hostname, data_size):
        """
        This is the first method that a new worker should call.

        Registers worker with coordinator, storing a WorkerInfo object in the
        workers dictionary with the IP address of the worker as the key.

        The WorkerInfo object establishes a ServerProxy connection to the worker,
        allowing the coordinator to ping it and ensure a functional two way connection.
        """

        print("\t[FROM " + hostname + "] Accepting new connection")
        # print("Accepting new connection on coordinator from " + hostname)
        worker = WorkerInfo(hostname, data_size)
        self.workers[hostname] = worker
        try:
            worker.notify("Ping")
        except Exception as e:
            print(f"Error pinging worker: {e}")

        # Let worker know if they should report test resuts
        return self.testing

    def send_update(self, hostname):
        """
        Send updated model weights, current epoch, and number of
        assigned local epochs to worker upon request
        """
        self.workers[hostname].last_pull = self.epoch
        raw_weights = [tensor.tolist() for tensor in self.weights]
        self.workers[hostname].epoch_start_time = time.time()
        return raw_weights, self.epoch, self.workers[hostname].num_epochs

    def receive_update(self, hostname, weights, epoch_completed, accuracy=None):
        """Receive update from worker"""

        # Check that worker is registered
        if hostname not in self.workers:
            return "Error: Worker not registered"

        self.workers[hostname].epoch_end_time = time.time()
        self.workers[hostname].epoch_durations.append(
            (
                epoch_completed,
                round(
                    self.workers[hostname].epoch_end_time
                    - self.workers[hostname].epoch_start_time,
                    3,
                ),
            )
        )
        # @TODO Alex please review and make sure this is ok with your load balancing
        # Check if this update is for current epoch
        if epoch_completed != self.epoch:
            self.workers[hostname].last_push = epoch_completed
            self.workers[hostname].num_epochs = max(
                1, (self.workers[hostname].num_epochs // 2)
            )
            print(
                f"\t[FROM: {hostname}] Recieved updated weights for epoch {epoch_completed}.\n \tCurrent epoch is {self.epoch}. Discarding outdated weights."
            )
            return "Discarding outdated update from " + hostname

        # Check that weights are the correct shape
        weights = [torch.FloatTensor(element) for element in weights]
        if not is_equal_dimensions(weights, self.weights):
            return "Error: Incorrect shape for weights"

        # Add update to queue, and update worker state
        self.updates[hostname] = weights
        self.workers[hostname].last_push = self.epoch
        if accuracy:
            self.epoch_accuracies[hostname] = accuracy

        # @TODO improve load balancing. If no workers are excluded by the quorum protocol, this will continue
        # incrementing for each epoch
        self.workers[hostname].num_epochs += 1  # Assign more work if finished early

        print(
            f"\t[FROM {hostname}] Recieved updated weights for epoch "
            + str(epoch_completed)
        )

        print(
            "\t\t"
            + str(len(self.updates))
            + " of "
            + str(len(self.workers))
            + " recieved. Current response rate: "
            + str(round(len(self.updates) / len(self.workers), 2))
        )

        # Start new epoch if enough updates have been received
        if len(self.updates) >= self.quorum_pct * len(self.workers):
            print(
                "Quorum achieved - ending epoch "
                + str(self.epoch)
                + " with "
                + str(len(self.updates))
                + " updates from "
                + str(len(self.workers))
                + " workers."
            )
            self.epoch_end_time = time.time()
            print(
                f"Epoch completed in {self.epoch_end_time - self.epoch_start_time} seconds."
            )
            self.start_new_epoch()

        return "Ok"

    def start_new_epoch(self):
        """Update global weights and start new epoch"""

        past_epoch = self.epoch
        self.epoch += 1
        past_updates = self.updates
        self.updates = {}
        past_epoch_accuracies = self.epoch_accuracies
        self.epoch_accuracies = {}

        # Merge updates into global weights, weighted average across list of tensors
        total_data = sum(self.workers[hostname].data_size for hostname in past_updates)
        for i in range(len(self.weights)):
            weighted_sum = torch.zeros_like(self.weights[i])
            for host, update in past_updates.items():
                weighted_sum += update[i] * (self.workers[host].data_size / total_data)
            self.weights[i] = weighted_sum

        # If accuracies were requested, merge them and output
        if self.testing:
            weighted_accuracy = 0
            for host, accuracy in past_epoch_accuracies.items():
                weighted_accuracy += accuracy * (
                    self.workers[host].data_size / total_data
                )
            self.accuracies.append(weighted_accuracy)
            print(f"Epoch Accuracy: {weighted_accuracy}")

        # if self.epoch >= self.max_epochs + 1:
        if self.epoch > self.max_epochs:
            # Shutdown server if we've reached max epochs
            print("Training complete")
            print(f"Accuracies: {self.accuracies}")
            # print("Final weights: ", self.weights)
            print("Epoch durations:")
            for worker in self.workers.values():
                # print("\t" + worker.hostname)
                print(worker.epoch_durations)

            for worker in self.workers.values():
                try:
                    worker.notify("Shutdown")
                except Exception as e:
                    print("Error shutting down " + worker.hostname)

            self.server.quit = True
            return

        # Notify workers of new epoch
        print(f"Starting epoch {self.epoch}. Sending notifications to workers")

        self.epoch_start_time = time.time()
        for worker in self.workers.values():
            try:
                worker.notify("Update Ready")
            except Exception as e:
                print("Error notifying " + worker.hostname + " of new epoch")

            # If worker never even responded to the previous notification, remove it
            # if worker.last_pull != self.epoch:
            if self.epoch - worker.last_pull > 2:
                print(
                    "Deleting "
                    + worker.hostname
                    + " from workers: no response to previous notification."
                )
                del self.workers[worker]

            # If worker never updated, decrease work load
            # if worker.last_push != self.epoch:
            if worker.last_push != past_epoch:
                worker.num_epochs = max(1, worker.num_epochs // 2)

    def handle_disconnect(self, hostname):
        """Remove worker from list of active workers"""
        del self.workers[hostname]
        return "Removed " + hostname + " from active worker list."

    def update_ready(self, hostname, worker_epoch):
        if worker_epoch != self.epoch:
            # print(
            #     f"[FROM {hostname}] Update ready? Yes. Worker epoch {worker_epoch} != global epoch {self.epoch}"
            # )
            return "Yes"
        return "No"

    def run(self):
        self.server.register_function(self.accept_connection, "connect")
        self.server.register_function(self.send_update, "get_update")
        self.server.register_function(self.receive_update, "load_update")
        self.server.register_function(self.handle_disconnect, "disconnect")
        self.server.register_function(self.update_ready, "update_ready")
        print("\n\nCoordinator serving at http://" + COORDINATOR_IP + ":" + str(PORT))
        self.server.serve_forever()


def main():
    coordinator = Coordinator(testing=True)
    coordinator.run()


if __name__ == "__main__":
    main()
