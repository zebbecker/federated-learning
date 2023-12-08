for server in $(cat ../serverlist.txt)
do 
  echo $server
  ssh -i ~/.ssh/$(whoami)-keypair.pem $(whoami)@$server 'pip3 install torchsummary'
  echo done
done 