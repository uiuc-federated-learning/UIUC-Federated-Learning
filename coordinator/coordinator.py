import logging
import pickle
import warnings
import grpc
import threading
import json
import torch

from src.model_aggregator import ModelAggregator
from src.flag_parser import Parser
import federated_pb2
import federated_pb2_grpc


def iterate_global_model(aggregator, remote_addresses, ports):
    remote_addresses = ["localhost:" + str(port) for port in ports] if remote_addresses == [] else remote_addresses
    print(remote_addresses)
    for epoch in range(parameters['global_epochs']):
        thread_list = []
        for i in range(len(remote_addresses)):
            thread = threading.Thread(target=train_hospital_model, args=(remote_addresses[i], aggregator, remote_addresses))
            thread_list.append(thread)
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
    
        aggregator.aggregate()
        print("Completed epoch %d. Aggregated all model weights." % (epoch))
    
    print('Completed all epochs.')

def train_hospital_model(hospital_address, aggregator, all_addresses):
    channel = grpc.insecure_channel(hospital_address)
    stub = federated_pb2_grpc.HospitalStub(channel)
    
    initReq = federated_pb2.InitializeReq(selfsocketaddress=hospital_address, allsocketaddresses=all_addresses, parameters=json.dumps(parameters))
    stub.Initialize(initReq)

    some_model = aggregator.global_model
    torch.save(some_model, "../globalmodel.pt")
    
    hospital_model = stub.ComputeUpdatedModel(federated_pb2.Model(model_obj=pickle.dumps(aggregator.global_model)))

    aggregator.add_hospital_data(pickle.loads(hospital_model.model.model_obj), hospital_model.training_samples)
    print("Received a set of weights from address: " + hospital_address)

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
