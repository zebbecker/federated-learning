from xmlrpc.server import SimpleXMLRPCServer

SERVER_NAME = "name of server?"
COORDINATOR_IP = "0.0.0.0"
PORT = 8000


def ping():
    return "ping!"


# Send training task to a worker
def get_work():
    pass


# Send training data to a worker (should only be called once)
def get_data():
    pass


# Allows workers to check if they should shut down.
def is_done():
    return True


server = SimpleXMLRPCServer((SERVER_NAME, PORT))
print("Coordinator started, listening on port", PORT)
server.register_function(ping, "ping")
server.register_function(get_work, "get_work")
server.register_function(get_data, "get_data")
server.register_function(is_done, "is_done")
server.serve_forever()
