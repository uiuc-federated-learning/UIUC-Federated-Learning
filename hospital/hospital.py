from concurrent import futures
import logging
import grpc
import numpy as np
import secrets
import pickle
import torch

from randomgen import AESCounter
from numpy.random import Generator

import federated_pb2
import federated_pb2_grpc

from src.flag_parser import Parser
from src.model_trainer import ModelTraining


import warnings
warnings.filterwarnings("ignore")

parser = Parser()
parameters = parser.parse_arguments()

def shift_weights(state_dict, shift_amount):
    power = (1<<shift_amount)

    for key, value in state_dict.items():
        if "bias" not in key:
            new_tensor = torch.zeros(value.shape, dtype=torch.int64)
            for row in range(value.shape[0]):
                for col in range(value.shape[1]):
                    new_tensor[row][col] = state_dict[key][row][col]*power
        else:
            new_tensor = torch.zeros(value.shape, dtype=torch.int64)
            for b in range(len(value)):
                new_tensor[b] = state_dict[key][b]*power
        state_dict[key] = new_tensor

def mask_weights(local_model_obj, positive_keys, negative_keys):
    for multiplier, keys in ((1, positive_keys), (-1, negative_keys)):
        for key in keys:
            aes_ctr = Generator(AESCounter(key))
            for layer_name in local_model_obj.keys():
                random_mask = aes_ctr.integers(-2**62, 2**62, local_model_obj[layer_name].shape)
                local_model_obj[layer_name] += multiplier * random_mask

class Hospital(federated_pb2_grpc.HospitalServicer):
    def __init__(self):
        self.positive_keys = []
        self.negative_keys = []

    def Initialize(self, intialize_req, context):
        print('Initialize called')
        for hospital_addr in intialize_req.allsocketaddresses:
            if hospital_addr > intialize_req.selfsocketaddress:
                channel = grpc.insecure_channel(hospital_addr)
                stub = federated_pb2_grpc.HospitalStub(channel)
                shared_key_resp = stub.FetchSharedKey(federated_pb2.FetchSharedKeyReq())
                shared_key = int(shared_key_resp.key)
                self.positive_keys.append(shared_key)
        
        return federated_pb2.InitializeResp()
        
    def FetchSharedKey(self, fetch_shared_key_req, context):
        shared_key = secrets.randbits(32)
        self.negative_keys.append(shared_key)
        return federated_pb2.FetchSharedKeyResp(key=str(shared_key))

    def ComputeUpdatedModel(self, global_model, context):
        local_model_obj, train_samples = model_trainer.ComputeUpdatedModel(global_model.model_obj)

        shift_weights(local_model_obj, parameters['shift_amount'])
        mask_weights(local_model_obj, self.positive_keys, self.negative_keys)
        print(local_model_obj)

        local_model = federated_pb2.TrainedModel(model=federated_pb2.Model(model_obj=pickle.dumps(local_model_obj)), training_samples=train_samples)
        
        return local_model

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_HospitalServicer_to_server(Hospital(), server)
    port = parameters['port']
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print("Serving")
    server.wait_for_termination()

if __name__ == "__main__":
    model_trainer = ModelTraining(parameters)
    logging.basicConfig()
    serve()