# kill process on workers and coordinator
echo -e "\n>>> Shutting down workers.\n"

for SERVER in 54.80.79.133 3.208.1.134 54.84.47.69 18.209.22.182; do
  ssh -i ~/.ssh/zbecker2-keypair.pem zbecker2@$SERVER /bin/bash << EOF
    killall python3
EOF
done

echo -e "\n>>> Finished shutting down workers.\n"