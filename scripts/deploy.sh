# Run from within scripts directory
echo "Starting deployment." 
scp -i ~/.ssh/$(whoami)-keypair.pem ../src/coordinator.py $(whoami)@15.156.205.154:~
echo "Finished copying coordinator program."
echo "Copied worker program to:"
for server in $(cat ../serverlist.txt)
do 
  scp -i ~/.ssh/$(whoami)-keypair.pem ../src/worker.py ../src/worker_model.py ../worker_requirements.txt $(whoami)@$server:~ &
  echo $server
done 
echo "Finished copying worker program.\n"