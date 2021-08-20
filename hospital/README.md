## Hospital Instructions

### Warning

In the current version of the code, the initial pairwise seed exchange is not encrypted or secured

### One Time Instructions 

Instructions are for brand new Azure VMs running Linux (Ubuntu 20.04), however should generalize fairly well for all setups

 1. Run `sudo apt-get update`
 2. Run `sudo apt-get install python3-pip` to install pip for our packages
 3. Navigate to the `hospital` folder
 4. Run `python3 -m pip install -r requirements.txt` to install the requirements
 5. Add image data to `/data/train/` and `/data/test/` as necessary 
	 - Positive cases go in `positive`
	 - Negative cases go in `negative`

### Start Hospital Program
1. Run `python3 hospital.py --port <PORT_NUMBER>` where `<PORT_NUMBER>` is any port accessible by the coodinator (8001 suggested)
	 - Alternatively set port in `system_config.json`

### Accessing PyTorch Models

 1. All the models are saved in `checkpoints`, organized by date, global epoch, and pre or post finetuned
