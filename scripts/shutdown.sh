# kill process on workers and coordinator
echo ">>> Shutting down workers."

for server in $(cat ../serverlist.txt)
do 
  ssh -i ~/.ssh/$(whoami)-keypair.pem $(whoami)@$server 'pkill -u $(whoami) python3'
done 

echo ">>> Finished shutting down workers."