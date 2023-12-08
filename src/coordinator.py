from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from xmlrpc.client import ServerProxy

import numpy as np
import torch

import worker_model

QUORUM_PERCENTAGE = 0.75

"""
Idea here is that server calls setup() to initialize model, and from there
out is primarily driven by workers making calls it its methods. 

When a certain threshold of updates have been recieved (we should play around
with how we define the threshold), the next worker that calls get_task will trigger 
the end of the current epoch. The coordinator will update the global model weights 
and declare a new epoch. 

In this design, the worker essentially loops. 
First, it calls server.get_task(worker_epoch), where worker_epoch is the ID of the 
most recent epoch the worker has finished training for. If worker just finished 
its epoch 3 task and the coordinator has not yet moved on the epoch 4, this will 
return some sort of null value. The worker should wait for some amount of time 
and try again later. Eventually, the worker will recieve a new set of weights with 
the next epoch ID. 

When it does, the worker will train, then call server.load_update(). 
This will send the weights and epoch id to the coordinator. If the coordinator
has already moved on from that epoch (i.e. this worker is a straggler), the weights 
will be discarded. Otherwise, they will be added to the update queue. If they are and
this worker pushes us across the threshold for starting a new epoch, the coordinator 
declares a new epoch. 

"""

COORDINATOR_IP = "139.140.215.220"  # Zeb
# COORDINATOR_IP = "139.140.197.180"
# COORDINATOR_IP = "hopper.bowdoin.edu"
PORT = 8082
# <- pass in as command line parameter: number of workers for quorum.
# Should be less than total number of workers to allow for fault tolerance
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

    def ping(self):
        """Ping worker to check connection"""
        return self.server.ping()

    def notify(self):
        """Notify worker that update is ready"""
        return self.server.notify()


class Coordinator:
    def __init__(self, quorum_percentage=QUORUM_PERCENTAGE, max_epochs=10):
        # Initialize weights using worker_model spec, list of tensors
        model = worker_model.model
        self.weights = [param.data for param in model.parameters()]

        # Set up RPC server
        self.server = SimpleXMLRPCServer(
            (COORDINATOR_IP, PORT), requestHandler=SimpleXMLRPCRequestHandler
        )

        # State for workers, updates, and training
        self.workers = {}
        self.updates = []
        self.quorum_pct = quorum_percentage
        self.epoch = 0
        self.max_epochs = max_epochs

    def accept_connection(self, hostname):
        """First call from worker, used to register worker with coordinator

        Allows worker to get set up with same structure as coordinator. Model
        spec format is TBD, but should probably be a dict of some sort that
        includes model architecture and hyperparameters.
        """
        print("Accepting connection on coordinator from " + hostname)
        worker = WorkerInfo(hostname)
        self.workers[hostname] = worker
        try:
            worker.ping()
        except Exception as e:
            print("Error pinging worker")
            return "Error: Worker not responding"
        return "connected!"

    def send_update(self, hostname):
        """Send update to worker upon request"""
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
        self.workers[hostname].num_epochs += 1  # Assign more work if finished early

        # Start new epoch if enough updates have been received
        if len(self.updates) > self.quorum_pct * len(self.workers):
            print("Coordinator starting new epoch")
            self.start_new_epoch()

        return "Ok"

    def start_new_epoch(self):
        """Update global weights and start new epoch"""

        # Merge updates into global weights, average across list of tensors
        for i in range(len(self.weights)):
            tensor_stack = torch.stack([update[i] for update in self.updates])
            self.weights[i] = torch.mean(tensor_stack, dim=0)

        # Notify workers of new epoch
        for worker in self.workers.values():
            try:
                worker.notify()
            except Exception as e:
                print("Error notifying worker of new epoch")

            # If worker never even responded to the notification, remove it
            if worker.last_pull != self.epoch:
                del self.workers[worker]

            # If worker never updated, decrease work load
            if worker.last_push != self.epoch:
                worker.num_epochs = max(1, worker.num_epochs // 2)

        # Reset updates and increment epoch
        self.updates = []
        self.epoch += 1

        # Shutdown server if we've reached max epochs
        if self.epoch > self.max_epochs:
            self.server.shutdown()  # Not sure if this is the right way to do this
            print("Training complete")
            print("Final weights: ", self.weights)

    def handle_disconnect(self, hostname):
        """Remove worker from list of active workers"""
        del self.workers[hostname]
        return "Removed " + hostname + " from active worker list."

    def run(self):
        self.server.register_function(self.accept_connection, "connect")
        self.server.register_function(self.send_update, "get_update")
        self.server.register_function(self.receive_update, "load_update")
        self.server.register_function(self.handle_disconnect, "disconnect")
        print("Coordinator serving at http://" + COORDINATOR_IP + "/" + str(PORT))
        self.server.serve_forever()


def main():
    coordinator = Coordinator()
    coordinator.run()


if __name__ == "__main__":
    main()
