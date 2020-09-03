from __future__ import print_function
import logging

import grpc

import helloworld_pb2
import helloworld_pb2_grpc

def run():
    # thread these functions
    ask_for_message('1234', 'one')
    ask_for_message('1235', 'two')
    ask_for_message('1236', 'three')

def ask_for_message(port, msg):
    channel = grpc.insecure_channel('localhost:' + port)
    stub = helloworld_pb2_grpc.GreeterStub(channel)
    response = stub.SayHello(helloworld_pb2.HelloRequest(name=msg))
    print("Greeter client received: " + response.message)
    channel.close()

logging.basicConfig()
run()