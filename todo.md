# TODO

### Worker
- Need to figure out data formatting, to make sure things are marshalled correctly
- Eventually change data to work with local files
- Eventually add a way to dynamically create model architecture

Alex/ML 
- How to tell clients what the training task is (e.g. do we define task in seperate file? What about path to data?)
- How to send weights back and forth with XMLRPC 
- how to combine weights at coordinator after each epoch
- training task and code 

Zeb/Infrastructure 
- add python virtual environment setup to deployment and start scripts 
- set var for username
- ensure needed python packages available on test machines
- Diagram worker-coordinator functions, system architecture 
- Implement structure (functions dont do anything)
- graceful shutdown of clients and coordinator 
- how to set quorum parameter? 
- dynamic quorum size
- investigate scalability of sequential server model. ThreadsMixIn? 

Presentation and Paper
- Design experiments 
- Presentation 