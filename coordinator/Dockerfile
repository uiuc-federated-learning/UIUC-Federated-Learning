# Start from 3.7.4 since latest points to 3.9 which has issues with torch
FROM python:3.7.4
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# We import 'coordinator.py' in '/coordinator/' folder.
ADD src /coordinator/src
ADD model_parameters.json /coordinator/
ADD coordinator.py /coordinator/

# This command changes the base directory of your image.
# Here we define '/coordinator/' as base directory.
WORKDIR /coordinator/

# We don't need to run any command because we will manually run the 
# coordinator script by sshing into it
