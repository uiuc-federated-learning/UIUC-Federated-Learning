from __future__ import print_function
import logging

import grpc

import helloworld_pb2
import helloworld_pb2_grpc

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:1234') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        response = stub.SayHello(helloworld_pb2.HelloRequest(name='I am inclined to think this works'))
    print("Greeter client received: " + response.message)

logging.basicConfig()
run()