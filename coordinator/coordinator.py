from __future__ import print_function
import logging

import grpc

import federated_pb2
import federated_pb2_grpc

from src.model_aggregator import ModelAggregator
from src.flag_parser import Parser

import pickle

import warnings

def iterate_global_model(aggregator, remote_addresses, ports):
    remote_addresses = ["localhost"]*len(ports) if remote_addresses == [] else remote_addresses
    assert len(remote_addresses) == len(ports)

    # TODO: thread these functions
    for epoch in range(parameters['global_epochs']):
        for i in range(len(remote_addresses)):
            address = remote_addresses[i] + ':' + ports[i]
            train_hospital_model(address)
    
        aggregator.aggregate()
        print("Completed epoch {epoch}. Aggregated all model weights.")
    
    print('Completed all epochs.')

def train_hospital_model(hospital_address):
    channel = grpc.insecure_channel(hospital_address)
    stub = federated_pb2_grpc.HospitalStub(channel)
    hospital_model = stub.ComputeUpdatedModel(federated_pb2.Model(weights=pickle.dumps(aggregator.global_model)))

    aggregator.add_hospital_data(pickle.loads(hospital_model.model.weights), hospital_model.training_samples)
    print("Received a set of weights")

    channel.close()

if __name__ == "__main__":
    # This prevents the following error message: "pickle support for Storage will be removed in 1.5. Use `torch.save` instead" from being printed to stdout. 
    warnings.filterwarnings("ignore") 
    logging.basicConfig()

    parser = Parser()
    parameters = parser.parse_arguments()

    aggregator = ModelAggregator(parameters)

    remote_addresses = parameters['remote_addresses']
    ports = parameters['ports']
    iterate_global_model(aggregator, remote_addresses, ports)
