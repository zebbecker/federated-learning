# kill process on workers and coordinator
echo ">>> Shutting down workers."

for server in $(cat serverlist.txt)
do 
  ssh -i ~/.ssh/zbecker2-keypair.pem zbecker2@$server 'pkill -u zbecker2 python3'
done 

echo ">>> Finished shutting down workers."