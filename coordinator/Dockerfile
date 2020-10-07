# Same thing than the 'server' Dockerfile.
FROM python:latest
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# Same thing than the 'server' Dockerfile.
# We import 'coordinator.py' in '/coordinator/' folder.
ADD src /coordinator/src
ADD model_parameters.json /coordinator/
ADD coordinator.py /coordinator/

# This command changes the base directory of your image.
# Here we define '/coordinator/' as base directory.
WORKDIR /coordinator/
