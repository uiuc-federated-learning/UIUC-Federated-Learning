# Same thing than the 'server' Dockerfile.
FROM python:latest
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# Same thing than the 'server' Dockerfile.
# We import 'client.py' in '/client/' folder.
ADD client.py /client/
ADD helloworld_pb2.py /client/
ADD helloworld_pb2_grpc.py /client/

# I would like to introduce something new, the 'WORKDIR' command.
# This command changes the base directory of your image.
# Here we define '/client/' as base directory.
WORKDIR /client/
