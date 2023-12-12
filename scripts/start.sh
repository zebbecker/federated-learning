# ssh into each worker machine, start worker program
# with appropriate command line args

echo ">>> Starting workers."

for server in $(cat ../fullserverlist.txt)
do 
  ssh -i ~/.ssh/$(whoami)-keypair.pem $(whoami)@$server "python3 worker.py 15.156.205.154:8082 $server" &
  echo ${server}
done  
echo  ">>> Finished starting workers."