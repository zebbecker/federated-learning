# ssh into each worker machine, start worker program
# with appropriate command line args (or seek them in a config file?)
# Store worker process ID in env var PID to use in shutdown script

# @TODO does not work when worker.py is long running process- script does not exit

echo -e "\n>>> Starting workers.\n"

SCRIPT='source env/bin/activate; nohup python3 worker.py; exit' 

for SERVER in 54.80.79.133 3.208.1.134 54.84.47.69 18.209.22.182; do
  ssh -i ~/.ssh/zbecker2-keypair.pem zbecker2@$SERVER "$SCRIPT"
done

echo -e "\n>>> Finished starting workers.\n"

# for SERVER in 54.80.79.133 3.208.1.134 54.84.47.69 18.209.22.182; do
#   ssh -i ~/.ssh/zbecker2-keypair.pem zbecker2@$SERVER /bin/bash << EOF
#     source env/bin/activate
#     python3 worker.py &
# EOF
# done

# echo -e "\n>>> Finished starting workers.\n"