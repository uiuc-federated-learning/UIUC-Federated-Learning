from concurrent import futures
import logging
import grpc
import numpy as np
import secrets
import pickle

import federated_pb2
import federated_pb2_grpc

from src.flag_parser import Parser
from src.model_trainer import ModelTraining

import warnings
warnings.filterwarnings("ignore")

parser = Parser()
parameters = parser.parse_arguments()

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
        
        return federated_pb2.InitializeResp()
        
    def FetchSharedKey(self, fetch_shared_key_req, context):
        shared_key = secrets.randbits(256)
        return federated_pb2.FetchSharedKeyResp(key=str(shared_key))

    def ComputeUpdatedModel(self, global_model, context):
        model, train_samples = model_trainer.ComputeUpdatedModel(global_model.model_obj)
        local_model = federated_pb2.TrainedModel(model=federated_pb2.Model(model_obj=pickle.dumps(model)), training_samples=train_samples)
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