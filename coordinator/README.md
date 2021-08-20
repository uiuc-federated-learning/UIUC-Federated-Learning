## Coordinator Instructions

### One Time Instructions 
Instructions are for brand new Azure VMs running Linux (Ubuntu 20.04), however should generalize fairly well for all setups

 1. Run `sudo apt-get update`
 2. Run `sudo apt-get install python3-pip` to install pip for our packages
 3. Run `python3 -m pip install -r requirements.txt` to install the requirements

### Start Coordinator Program
1. Modify the following variables as necessary
	- `system_config.conf`
		- `global_epochs` changes the number of global aggregation rounds 
	- `model_parameters.conf`
		- `local_lr` changes the hospitals' learning rates
		- `local_epochs` is the number of epochs the hospitals train
2. Run the program with the following format `python3 coordinator.py --remote_addresses  <ip 1>:<port> <ip 2>:<port> <ip 3>:<port> [--checkpoint <filepath>]` (Note: checkpoint arg is optional and best practice is to use full filepath)
	- Example command: `python3 coordinator.py --remote_addresses 52.188.12.164:8001 52.188.14.142:8001 52.255.194.72:8001 52.188.10.87:8001 --checkpoint ~/models/backup5.pth`

### Accessing PyTorch Models

 1. All the aggregated models are saved in `checkpoints`, organized by date and global epoch
