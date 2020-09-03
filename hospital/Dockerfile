# Just a remember, dockerfile must always start by importing the base image.
# We use the keyword 'FROM' to do that.
# In our example, we want to import the python image (from DockerHub).
# So, we write 'python' for the image name and 'latest' for the version.
FROM python:latest
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# In order to launch our python code, we must import the 'hospital.py' and 'index.html' file.
# We use the keyword 'ADD' to do that.
# Just a remember, the first parameter 'hospital.py' is the name of the file on the host.
# The second parameter '/hospital/' is the path where to put the file on the image.
# Here we put files at the image '/hospital/' folder.
ADD hospital.py /hospital/
ADD helloworld_pb2.py /hospital/
ADD helloworld_pb2_grpc.py /hospital/

# I would like to introduce something new, the 'WORKDIR' command.
# This command changes the base directory of your image.
# Here we define '/hospital/' as base directory (where all commands will be executed).
WORKDIR /hospital/