from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from xmlrpc.client import ServerProxy

import numpy as np
import torch

import worker_model

# Usage: python3 coordinator.py
# (set QP, coordinator IP, and port number in script below)

# @TODO add output if we are unable to achieve quorum percentage.
QUORUM_PERCENTAGE = 0.75

COORDINATOR_IP = "hopper.bowdoin.edu"
PORT = 8082

WORKER_STARTING_EPOCHS = 4


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

    def __init__(self, hostname, num_epochs=WORKER_STARTING_EPOCHS):
        self.server = ServerProxy(hostname)
        self.num_epochs = num_epochs
        self.last_push = -1
        self.last_pull = -1
        self.hostname = hostname

    def ping(self):
        """Ping worker to check connection"""
        return self.server.ping()

    def notify(self):
        """Notify worker that update is ready"""
        return self.server.notify()


# @TODO when we reach max epochs, print something or shutdown gracefully- currently just hangs
class Coordinator:
    def __init__(self, quorum_percentage=QUORUM_PERCENTAGE, max_epochs=10):
        # Initialize weights using worker_model spec, list of tensors
        model = worker_model.model
        self.weights = [param.data for param in model.parameters()]

        # Set up RPC server
        self.server = SimpleXMLRPCServer(
            (COORDINATOR_IP, PORT),
            requestHandler=SimpleXMLRPCRequestHandler,
            logRequests=False,
        )

        # State for workers, updates, and training
        self.workers = {}
        self.updates = []
        self.quorum_pct = quorum_percentage
        self.epoch = 0
        self.max_epochs = max_epochs

    def accept_connection(self, hostname):
        """
        This is the first method that a new worker should call.

        Registers worker with coordinator, storing a WorkerInfo object in the
        workers dictionary with the IP address of the worker as the key.

        The WorkerInfo object establishes a ServerProxy connection to the worker,
        allowing the coordinator to ping it and ensure a functional two way connection.
        """

        print("[FROM " + hostname + "] Accepting new connection")
        # print("Accepting new connection on coordinator from " + hostname)
        worker = WorkerInfo(hostname)
        self.workers[hostname] = worker
        try:
            worker.ping()
        except Exception as e:
            print("Error pinging worker")
            return "Error: Worker not responding"
        return "connected!"

    def send_update(self, hostname):
        """
        Send updated model weights, current epoch, and number of
        assigned local epochs to worker upon request
        """
        self.workers[hostname].last_pull = self.epoch
        raw_weights = [tensor.tolist() for tensor in self.weights]
        return raw_weights, self.epoch, self.workers[hostname].num_epochs

    def receive_update(self, hostname, weights):
        """Receive update from worker"""

        # Check that worker is registered
        if hostname not in self.workers:
            return "Error: Worker not registered"

        # Check that weights are the correct shape
        weights = [torch.FloatTensor(element) for element in weights]
        if not is_equal_dimensions(weights, self.weights):
            return "Error: Incorrect shape for weights"

        # Add update to queue, and update worker state
        self.updates.append(weights)
        self.workers[hostname].last_push = self.epoch

        # @TODO improve load balancing. If no workers are excluded by the quorum protocol, this will continue
        # incrementing for each epoch
        self.workers[hostname].num_epochs += 1  # Assign more work if finished early

        print(
            "[FROM:"
            + hostname
            + "] Recieved updated weights for epoch "
            + str(self.workers[hostname].last_push)
        )

        print(
            str(len(self.updates))
            + " of "
            + str(len(self.workers))
            + " recieved. Current response rate: "
            + str(len(self.updates) / len(self.workers))
        )

        # Start new epoch if enough updates have been received
        if len(self.updates) > self.quorum_pct * len(self.workers):
            print(
                "Quorum achieved - ending epoch "
                + str(self.epoch)
                + " with "
                + str(len(self.updates))
                + " updates from "
                + str(len(self.workers))
                + " workers."
            )
            self.start_new_epoch()

        return "Ok"

    def start_new_epoch(self):
        """Update global weights and start new epoch"""

        # Merge updates into global weights, average across list of tensors
        for i in range(len(self.weights)):
            tensor_stack = torch.stack([update[i] for update in self.updates])
            # @TODO implement weighted means: workers with more data should count more
            self.weights[i] = torch.mean(tensor_stack, dim=0)

        print(
            "\nStarting "
            + str(self.epoch + 1)
            + ". Sending updated weights and tasks to all workers."
        )

        # Notify workers of new epoch
        for worker in self.workers.values():
            try:
                worker.notify()
            except Exception as e:
                print("Error notifying " + worker.hostname + " of new epoch")

            # If worker never even responded to the notification, remove it
            if worker.last_pull != self.epoch:
                del self.workers[worker]

            # If worker never updated, decrease work load
            if worker.last_push != self.epoch:
                worker.num_epochs = max(1, worker.num_epochs // 2)

        # Reset updates and increment epoch
        self.updates = []
        self.epoch += 1

        # @TODO
        # Shutdown server if we've reached max epochs
        if self.epoch > self.max_epochs:
            print("Training complete")
            print("Final weights: ", self.weights)
            self.server.shutdown()  # Not sure if this is the right way to do this

    def handle_disconnect(self, hostname):
        """Remove worker from list of active workers"""
        del self.workers[hostname]
        return "Removed " + hostname + " from active worker list."

    def run(self):
        self.server.register_function(self.accept_connection, "connect")
        self.server.register_function(self.send_update, "get_update")
        self.server.register_function(self.receive_update, "load_update")
        self.server.register_function(self.handle_disconnect, "disconnect")
        print("Coordinator serving at http://" + COORDINATOR_IP + ":" + str(PORT))
        self.server.serve_forever()


def main():
    coordinator = Coordinator()
    coordinator.run()


if __name__ == "__main__":
    main()
