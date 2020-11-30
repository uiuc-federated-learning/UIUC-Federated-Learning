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

from randomgen import AESCounter
from numpy.random import Generator

import federated_pb2
import federated_pb2_grpc

from src.flag_parser import Parser
from src.model_trainer import ModelTraining

from pprint import pprint

import warnings
warnings.filterwarnings("ignore")

MAX_MESSAGE_LENGTH = 1000000000 # 1GB maximum model size (gRPC message size)


def shift_weights(state_dict, shift_amount):
    power = (1<<shift_amount)

    for key, value in state_dict.items():
        new_tensor = torch.zeros(value.shape, dtype=torch.int64)
        dims_to_evaluate = [list(range(dim)) for dim in value.shape]
        tups = [x for x in itertools.product(*dims_to_evaluate)]
        for tup in tups:
            new_tensor[tup] = state_dict[key][tup]*power
        
        state_dict[key] = new_tensor

# Even though the MIN_INT for torch.int64 is actually -2**63, we need to generate an
# integer x such that -x is also a 64 bit integer. -(-2**63) = 2**63, which does not
# fit in a 64 bit number.
MIN_GENERATEABLE_INT = -2**63 + 1
MAX_GENERATEABLE_INT =  2**63 - 1

def mask_weights(local_model_obj, positive_keys, negative_keys):
    for multiplier, keys in ((1, positive_keys), (-1, negative_keys)):
        for key in keys:
            aes_ctr = Generator(AESCounter(key))
            for layer_name in local_model_obj.keys():
                random_mask = aes_ctr.integers(MIN_GENERATEABLE_INT, MAX_GENERATEABLE_INT, local_model_obj[layer_name].shape, dtype=np.int64, endpoint=True)
                local_model_obj[layer_name] += multiplier * random_mask

class Hospital(federated_pb2_grpc.HospitalServicer):
    def __init__(self):
        self.positive_keys = []
        self.negative_keys = []
        self.parameters = dict()

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
        
        return federated_pb2.InitializeResp()
        
    def FetchSharedKey(self, fetch_shared_key_req, context):
        shared_key = secrets.randbits(32)
        self.negative_keys.append(shared_key)
        return federated_pb2.FetchSharedKeyResp(key=str(shared_key))

    def ComputeUpdatedModel(self, global_model, context):
        print('ComputeUpdatedModel called')
        # Load ScriptModule from io.BytesIO object
        # global_model = pickle.loads(global_model.model_obj)
        buffer = io.BytesIO(global_model.traced_model)
        buffercopy = copy.deepcopy(buffer)
        global_model_loaded_jit = torch.jit.load(buffer)

        local_model_obj, train_samples = self.model_trainer.ComputeUpdatedModel(global_model_loaded_jit, buffercopy)

        shift_weights(local_model_obj, self.parameters['shift_amount'])
        mask_weights(local_model_obj, self.positive_keys, self.negative_keys)

        local_model = federated_pb2.TrainedModel(model=federated_pb2.Model(model_obj=pickle.dumps(local_model_obj)), training_samples=train_samples)
        
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
    print("Serving")
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()