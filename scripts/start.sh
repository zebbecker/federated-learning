# ssh into each worker machine, start worker program
# with appropriate command line args (or seek them in a config file?)

function start_server() 
{
  ssh -i ~/.ssh/zbecker2-keypair.pem zbecker2@$1 'python3 worker.py &'
}

echo ">>> Starting workers."

for server in $(cat serverlist.txt)
do 
  start_server ${server} &
done  
echo  ">>> Finished starting workers."