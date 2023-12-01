from xmlrpc.server import SimpleXMLRPCServer

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

SERVER_NAME = "name of server?"
COORDINATOR_IP = "0.0.0.0"
PORT = 8000
NUM_WORKERS = 10
# <- pass in as command line parameter: number of workers for quorum.
# Should be less than total number of workers to allow for fault tolerance
QUORUM = 8
MAX_EPOCHS = 100  # @TODO add graceful shutdown for workers and coordinator
# Not needed now- can just use shutdown script.

epoch = 0
updates = []
weights = []
workers_completed = 0


def ping():
    return "ping!"


# Send current weights and epoch number to worker ([], int)
def get_task(worker_epoch):
    if worker_epoch < epoch:
        return (weights, epoch)
    else:
        # if worker has already completed training for this epoch, don't bother sending current weights over network again
        return (None, None)


# @TODO graceful shutdown
# Allows workers to check if they should shut down.
def is_done():
    return False


def load_update(update, worker_epoch):
    if worker_epoch != epoch:
        # update outdated: discard
        return
    else:
        updates.append(update)
        workers_completed += 1

    if workers_completed > QUORUM:
        start_new_epoch()


def setup():
    # do any model setup needed here
    # define first round weights here
    pass


def apply_updates():
    # Update model weights here
    pass


def start_new_epoch():
    global workers_completed, weights, epoch  # modify global vars
    workers_completed = 0
    weights = apply_updates(updates)
    epoch += 1


server = SimpleXMLRPCServer((SERVER_NAME, PORT))
print("Coordinator started, listening on port", PORT)
server.register_function(ping, "ping")
server.register_function(get_task, "get_task")
server.register_function(load_update, "load_update")
server.register_function(is_done, "is_done")

setup()  # initialize model and set epoch to 0

server.serve_forever()
