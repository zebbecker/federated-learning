import xmlrpc.client
import sys


def train(data):
    print("Training with data", data)


commands = {"train": train}


def main():
    hostname = sys.argv[1]
    name = "http://" + hostname
    server = xmlrpc.client.ServerProxy(name)

    print("Attempting to connect to", name, "...")

    try:
        server.Coordinator.ping()
        print("Connected to", hostname)
    except ConnectionRefusedError:
        print("Error: Unable to connect to", hostname)
        sys.exit()

    while True:
        # data = server.getWork()
        # ^  can we get this to block for a bit? Or we can just put a sleep

        # weights = train(data)
        # server.sendResults(weights)
        # if server.isDone(): return
        pass


if __name__ == "__main__":
    main()
