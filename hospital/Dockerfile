# Start from 3.7.4 since latest points to 3.9 which has issues with torch
FROM python:3.7.4
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# Same thing than the 'server' Dockerfile.
# We import 'coordinator.py' in '/coordinator/' folder.
ADD src /hospital/src
ADD model_parameters.json /hospital/
ADD hospital.py /hospital/
ADD federated_pb2.py /hospital/
ADD federated_pb2_grpc.py /hospital/

# This command changes the base directory of your image.
# Here we define '/hospital/' as base directory.
WORKDIR /hospital/

# Now run the hospital on port 8000 on the machine.
CMD ["python", "hospital.py", "--port", "8000"]
