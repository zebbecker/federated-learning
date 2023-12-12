# ssh into each worker machine, start worker program
# with appropriate command line args (or seek them in a config file?)

function start_server() 
{
  ssh -i ~/.ssh/$(whoami)-keypair.pem $(whoami)@$1 "python3 worker.py 15.156.205.154:8082 $1 &"
}

echo ">>> Starting workers."

for server in $(cat ../fullserverlist.txt)
do 
  start_server ${server} &
  echo ${server}
done  
echo  ">>> Finished starting workers."