# Same thing than the 'server' Dockerfile.
FROM python:latest
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# Same thing than the 'server' Dockerfile.
# We import 'coordinator.py' in '/coordinator/' folder.
ADD src /coordinator/src
ADD config.json /coordinator/
ADD coordinator.py /coordinator/
ADD helloworld_pb2.py /coordinator/
ADD helloworld_pb2_grpc.py /coordinator/

# I would like to introduce something new, the 'WORKDIR' command.
# This command changes the base directory of your image.
# Here we define '/coordinator/' as base directory.
WORKDIR /coordinator/