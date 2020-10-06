from concurrent import futures
import logging

import grpc

import federated_pb2
import federated_pb2_grpc

import numpy as np

from src.flag_parser import Parser
from src.model_trainer import ModelTraining

import warnings
warnings.filterwarnings("ignore")

parser = Parser()
parameters = parser.parse_arguments()

class Hospital(federated_pb2_grpc.HospitalServicer):

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