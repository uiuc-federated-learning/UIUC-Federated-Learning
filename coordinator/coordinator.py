from __future__ import print_function
import logging

import grpc

import federated_pb2
import federated_pb2_grpc

import numpy as np
import copy

import torch
from torchvision import datasets, transforms

from src.sampling import iid, non_iid
from src.models import LR, MLP, CNNMnist
from src.utils import global_aggregate, network_parameters, test_inference
from src.ModelAggregator import ModelAggregator
from src.Parser import Parser

from collections import OrderedDict, Counter

from random import randint

import warnings
import os
import pickle
warnings.filterwarnings("ignore")

parser = Parser()
parameters = parser.parse_arguments()

aggregator = ModelAggregator(parameters)

def run():
    # thread these functions
    for _ in range(parameters['global_epochs']):
        send_model('1234')
    
        aggregator.aggregate()
        print("Aggregated weights")

def send_model(port):
    channel = grpc.insecure_channel('localhost:' + port)
    stub = federated_pb2_grpc.FederatedStub(channel)
    hospital_model = stub.GetUpdatedModel(federated_pb2.UpdatedModelRequest(global_model=pickle.dumps(aggregator.global_model)))
    aggregator.add_hospital_data(pickle.loads(hospital_model.weights), hospital_model.local_size)
    print("Received a set of weights")
    channel.close()

logging.basicConfig()
run()