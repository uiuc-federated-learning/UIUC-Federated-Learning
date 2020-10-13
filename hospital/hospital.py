from concurrent import futures
import logging
import grpc
import numpy as np
import secrets

import federated_pb2
import federated_pb2_grpc

from src.flag_parser import Parser
from src.model_trainer import ModelTraining

import warnings
warnings.filterwarnings("ignore")

parser = Parser()
parameters = parser.parse_arguments()

class Hospital(federated_pb2_grpc.HospitalServicer):
    shared_keys = []

    def Initialize(self, intialize_req):
        for hospital_addr in intialize_req.allsocketaddresses:
            if hospital_addr > intialize_req.selfsocketaddress:
                channel = grpc.insecure_channel(hospital_addr)
                stub = federated_pb2_grpc.HospitalStub(channel)
                shared_key_resp = stub.FetchSharedKey(federated_pb2.FetchSharedKeyReq())
                shared_keys.append(shared_key_resp.key)
        
    def FetchSharedKey(self, fetch_shared_key_req):
        shared_key = secrets.token_bytes(32)
        return federated_pb2.FetchSharedKeyResp(key=shared_key) 

    def ComputeUpdatedModel(self, global_model, context):
        model = local_model.ComputeUpdatedModel(global_model, context)
        print("Sending model")
        return model
            

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_HospitalServicer_to_server(Hospital(), server)
    port = parameters['port']
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print("Serving")
    server.wait_for_termination()

if __name__ == "__main__":
    local_model = ModelTraining(parameters)

    logging.basicConfig()
    serve()