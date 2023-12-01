# TODO

### Worker
- Right now Worker assumes that coordinator has connect and update methods
- Need to figure out data formatting, to make sure things are marshalled correctly
- Eventually change data to work with local files
- Eventually add a way to dynamically create model architecture

Alex/ML 
- How to tell clients what the training task is (e.g. do we define task in seperate file? What about path to data?)
- How to send weights back and forth with XMLRPC 
- how to combine weights at coordinator after each epoch
- training task and code 

Zeb/Infrastructure 
- Diagram worker-coordinator functions, system architecture 
- Implement structure (functions dont do anything)

Presentation and Paper
- Design experiments 
- Presentation 