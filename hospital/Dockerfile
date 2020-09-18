# Same thing than the 'server' Dockerfile.
FROM python:latest
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# Same thing than the 'server' Dockerfile.
# We import 'coordinator.py' in '/coordinator/' folder.
ADD src /hospital/src
ADD model_parameters.json /hospital/
ADD hospital.py /hospital/

# This command changes the base directory of your image.
# Here we define '/hospital/' as base directory.
WORKDIR /hospital/
