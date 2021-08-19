import logging
import pickle
import warnings
import grpc
import threading
import json
import torch
import io
from time import time
import copy
import os
from datetime import datetime

from src.model_aggregator import ModelAggregator
from src.flag_parser import Parser
from src.models import MLP
import federated_pb2
import federated_pb2_grpc
from datetime import datetime

MAX_MESSAGE_LENGTH = 1000000000 # 1GB maximum model size (message size)
INT_MAX = 2147483647
save_folder = ""

save_folder = ""

def iterate_global_model(aggregator, remote_addresses, ports):
    remote_addresses = ["localhost:" + str(port) for port in ports] if remote_addresses == [] else remote_addresses
    print("Remote addresses:", remote_addresses)

    if os.environ.get('https_proxy'):
        del os.environ['https_proxy']
    if os.environ.get('http_proxy'):
        del os.environ['http_proxy']

    thread_list = []
    for i in range(len(remote_addresses)):
        thread = threading.Thread(target=initialize_hospital, args=(i, remote_addresses[i], remote_addresses))
        thread_list.append(thread)
        thread.start()
    for thread in thread_list:
        thread.join()

    print("Done Init")

    for epoch in range(parameters['global_epochs']):
        thread_list = []
        print("")
        for i in range(len(remote_addresses)):
            thread = threading.Thread(target=train_hospital_model, args=(remote_addresses[i], aggregator.global_model, None, remote_addresses, epoch))
            thread_list.append(thread)
            thread.start()
        for thread in thread_list:
            thread.join()
    
        aggregator.aggregate()
        
        print("Completed epoch %d. Aggregated all model weights." % (epoch+1))
        torch.save(aggregator.global_model, f'./checkpoints/' + save_folder + f'/aggregated_{parameters["model"]}_epoch{epoch}.pth')
            
    print('Completed all epochs!')

def initialize_hospital(client_num, hospital_address, all_addresses):
    print("Attempting to connenct to", hospital_address)
    
    channel = grpc.insecure_channel(hospital_address, options=[('grpc.enable_http_proxy', 0)])
    stub = federated_pb2_grpc.HospitalStub(channel)

    new_params = copy.deepcopy(parameters)
    new_params['client_num'] = client_num
    initReq = federated_pb2.InitializeReq(selfsocketaddress=hospital_address, allsocketaddresses=all_addresses, parameters=json.dumps(new_params))
    stub.Initialize(initReq)

    channel.close()

def train_hospital_model(hospital_address, global_model, traced_model_bytes, all_addresses, epoch):
    # credentials = grpc.ssl_channel_credentials()
    channel = grpc.insecure_channel(hospital_address, options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ])
    stub = federated_pb2_grpc.HospitalStub(channel)
    
    print('Calling the gRPC endpoint for ' + hospital_address + ' epoch ' + str(epoch + 1))
    start = time()
    hospital_model = stub.ComputeUpdatedModel(federated_pb2.Model(model_obj=pickle.dumps(global_model), traced_model=None))
    aggregator.add_hospital_data(pickle.loads(hospital_model.model.model_obj), hospital_model.training_samples)
    end = time()
    print(f"Received a set of weights from {hospital_address}, took {end-start} seconds")
    channel.close()

if __name__ == "__main__":
    # This prevents the following error message: "pickle support for Storage will be removed in 1.5. Use `torch.save` instead" from being printed to stdout. 
    warnings.filterwarnings("ignore") 
    logging.basicConfig()

    parser = Parser()
    parameters = parser.parse_arguments()

    save_folder = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    if not os.path.exists(f'checkpoints/' + save_folder):
        os.makedirs(f'checkpoints/' + save_folder)

    aggregator = ModelAggregator(parameters)

    remote_addresses = parameters['remote_addresses']
    ports = parameters['ports']
    iterate_global_model(aggregator, remote_addresses, ports)
