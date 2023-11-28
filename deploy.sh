echo "Starting deployment." 
for server in $(cat serverlist.txt)
do 
  scp -i ~/.ssh/zbecker2-keypair.pem worker.py zbecker2@$server:~
done 
wait 
echo "Finished copying worker program.\n"