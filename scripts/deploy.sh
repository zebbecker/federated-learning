# Run from within scripts directory
echo "Starting deployment." 
for server in $(cat ../serverlist.txt)
do 
  scp -i ~/.ssh/$(whoami)-keypair.pem ../src/worker.py ../src/worker_model.py ../worker_requirements.txt $(whoami)@$server:~
done 
wait 
echo "Finished copying worker program.\n"