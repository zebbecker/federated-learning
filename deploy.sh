echo "Starting deployment." 
for server in $(cat serverlist.txt)
do 
  scp -i ~/.ssh/aracape-keypair.pem worker.py aracape@$server:~
done 
wait 
echo "Finished copying worker program.\n"