# Send updated code to worker nodes
for SERVER in 54.80.79.133 3.208.1.134 54.84.47.69 18.209.22.182; do
  scp -i ~/.ssh/zbecker2-keypair.pem worker.py zbecker2@$SERVER:~
done

# Send updated code to coordinator node 