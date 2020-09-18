from __future__ import print_function
import logging

import grpc

import federated_pb2
import federated_pb2_grpc

from src.model_aggregator import ModelAggregator
from src.flag_parser import Parser

import pickle

import warnings

def run():
    # TODO: thread these functions
    for _ in range(parameters['global_epochs']):
        send_model('8001')
        send_model('8002')
    
        aggregator.aggregate()
        print("Aggregated weights")

def send_model(port):
    channel = grpc.insecure_channel('localhost:' + port)
    stub = federated_pb2_grpc.HospitalStub(channel)
    hospital_model = stub.ComputeUpdatedModel(federated_pb2.UpdatedModelRequest(global_model=pickle.dumps(aggregator.global_model)))
    aggregator.add_hospital_data(pickle.loads(hospital_model.weights), hospital_model.local_size)
    print("Received a set of weights")
    channel.close()

if __name__ == "__main__":
    # This prevents the "pickle support for Storage will be removed in 1.5. Use `torch.save` instead" from being printed to stdout. 
    warnings.filterwarnings("ignore") 

    parser = Parser()
    parameters = parser.parse_arguments()

    aggregator = ModelAggregator(parameters)
    logging.basicConfig()
    run()