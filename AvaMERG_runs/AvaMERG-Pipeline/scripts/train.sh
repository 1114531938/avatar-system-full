echo "Starting training......"
nohup deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 merg_code/train.py 
