from concurrent import futures
import logging
import grpc
import numpy as np
import secrets
import pickle
import torch
import json
import io
import copy
import itertools
from time import time

from randomgen import AESCounter
from numpy.random import Generator
from datetime import datetime
import os

import federated_pb2
import federated_pb2_grpc

from src.flag_parser import Parser
from src.model_trainer import ModelTraining

from pprint import pprint

import warnings
warnings.filterwarnings("ignore")

MAX_MESSAGE_LENGTH = 1000000000 # 1GB maximum model size (message size)
BITS = 64

def shift_weights(state_dict, shift_amount):
    power = (1<<shift_amount)

    for key, value in state_dict.items():
        state_dict[key] = (state_dict[key].double()*power).long().cpu()
    
    

def mask_weights(local_model_obj, positive_keys, negative_keys):
    
    for multiplier, keys in ((1, positive_keys), (-1, negative_keys)):
        for key in keys:
            aes_ctr = Generator(AESCounter(key))
            for layer_name in local_model_obj.keys():
                random_mask = aes_ctr.integers(-(2**(BITS-1)), (2**(BITS-1)) - 1, local_model_obj[layer_name].shape)

                local_model_obj[layer_name] = local_model_obj[layer_name].long().cpu()
                
                local_model_obj[layer_name] += multiplier * random_mask

                
class Hospital(federated_pb2_grpc.HospitalServicer):
    def __init__(self):
        self.positive_keys = []
        self.negative_keys = []
        self.parameters = dict()
        self.save_folder = ""
        self.global_epoch = 0

    def Initialize(self, intialize_req, context):
        print('Initialize called')
        parser = Parser()
        self.parameters = parser.parse_arguments(intialize_req.parameters)

        self.model_trainer = ModelTraining(self.parameters)

        for hospital_addr in intialize_req.allsocketaddresses:
            if hospital_addr > intialize_req.selfsocketaddress:
                channel = grpc.insecure_channel(hospital_addr)
                stub = federated_pb2_grpc.HospitalStub(channel)
                shared_key_resp = stub.FetchSharedKey(federated_pb2.FetchSharedKeyReq())
                shared_key = int(shared_key_resp.key)
                self.positive_keys.append(shared_key)
        
        self.save_folder = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        if not os.path.exists(f'checkpoints/' + self.save_folder):
            os.makedirs(f'checkpoints/' + self.save_folder)
    
        self.global_epoch = 0

        return federated_pb2.InitializeResp()
        
    def FetchSharedKey(self, fetch_shared_key_req, context):
        shared_key = secrets.randbits(32)
        self.negative_keys.append(shared_key)
        return federated_pb2.FetchSharedKeyResp(key=str(shared_key))

    def ComputeUpdatedModel(self, global_model, context):
        print('\n\nStarting global epoch', self.global_epoch+1)
        
        global_model = pickle.loads(global_model.model_obj)

        torch.save(global_model, f'./checkpoints/' + self.save_folder + f'/prefinetuned_epoch{self.global_epoch+1}.pth')
        
        local_model_obj, train_samples = self.model_trainer.ComputeUpdatedModel(global_model, None)

        print('Shifting weights')
        shift_weights(local_model_obj, self.parameters['shift_amount'])
        
        print('Masking weights')
        mask_weights(local_model_obj, self.positive_keys, self.negative_keys)
        
        local_model = federated_pb2.TrainedModel(model=federated_pb2.Model(model_obj=pickle.dumps(local_model_obj)), training_samples=train_samples)
        
        torch.save(local_model, f'./checkpoints/' + self.save_folder + f'/postfinetuned_epoch{self.global_epoch+1}.pth')

        self.global_epoch += 1

        print('Sending local model')
        return local_model

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ])

    federated_pb2_grpc.add_HospitalServicer_to_server(Hospital(), server)
    parser = Parser()
    parameters = parser.parse_arguments(open('system_config.json', 'r').read())
    port = parameters['port']
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print("Serving on port", port)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()
