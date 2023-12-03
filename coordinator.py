from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy

import numpy as np
import torch

import worker_model

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

COORDINATOR_IP = "139.140.197.180"
PORT = 8081
# <- pass in as command line parameter: number of workers for quorum.
# Should be less than total number of workers to allow for fault tolerance

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


class Coordinator:

    def __init__(self, quorum_percentage=0.8, max_epochs=10):

        # Initialize weights using worker_model spec, list of tensors
        model = worker_model.model
        self.weights = [param.data for param in model.parameters()]

        # Set up RPC server
        self.server = SimpleXMLRPCServer((COORDINATOR_IP, PORT))

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

        self.workers[hostname] = ServerProxy(hostname)

    def send_update(self):
        """Send update to worker upon request"""
        return [tensor.tolist() for tensor in self.weights], self.epoch

    def receive_update(self, weights):
        """Receive update from worker"""

        # Check that weights are the correct shape
        weights = [torch.FloatTensor(element) for element in weights]
        if not is_equal_dimensions(weights, self.weights):
            return "Error: Incorrect shape for weights"
        
        # Add update to queue, and start new epoch if enough updates have been received
        self.updates.append(weights)
        if len(self.updates) > self.quorum_pct * len(self.workers):
            self.start_new_epoch()

        return "Ok"

    def start_new_epoch(self):
        """Update global weights and start new epoch"""
        
        # Merge updates into global weights
        updates = np.array(self.updates)
        self.weights = np.mean(updates, axis=0)

        # Notify workers of new epoch
        for worker in self.workers.values():
            worker.notify()

        # Reset updates and increment epoch
        self.updates = []
        self.epoch += 1

        if self.epoch > self.max_epochs:
            self.server.shutdown()  # Not sure if this is the right way to do this
            print("Training complete")
            print("Final weights: ", self.weights)

    def handle_disconnect(self, hostname):
        """Remove worker from list of active workers"""
        del self.workers[hostname]

    def run(self):
        self.server.register_function(self.accept_connection, "connect")
        self.server.register_function(self.send_update, "get_update")
        self.server.register_function(self.receive_update, "load_update")
        self.server.register_function(self.handle_disconnect, "disconnect")
        self.server.serve_forever()


def main():
    coordinator = Coordinator()
    coordinator.run()


if __name__ == "__main__":
    main()
