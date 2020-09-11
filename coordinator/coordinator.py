from __future__ import print_function
import logging

import grpc

import federated_pb2
import federated_pb2_grpc

import numpy as np
import copy
import argparse
import json

import torch
from torchvision import datasets, transforms

from src.sampling import iid, non_iid
from src.models import LR, MLP, CNNMnist
from src.utils import global_aggregate, network_parameters, test_inference
from src.attacks import attack_updates
from src.defense import defend_updates
from src.ModelAggregator import ModelAggregator

from collections import OrderedDict, Counter

from random import randint

import warnings
import os
import pickle
warnings.filterwarnings("ignore")

############################## Reading Arguments ##############################

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='', help="name of the experiment")
parser.add_argument('--seed', type=int, default=0, help="seed for running the experiments")
parser.add_argument('--data_source', type=str, default="MNIST", help="dataset to be used", choices=['MNIST'])
parser.add_argument('--sampling', type=str, default="iid", help="sampling technique for client data", choices=['iid', 'non_iid'])
parser.add_argument('--num_users', type=int, default=100, help="number of clients to create")
parser.add_argument('--num_shards_user', type=int, default=2, help="number of classes to give to the user")
parser.add_argument('--train_test_split', type=float, default=1.0, help="train test split at the client end")
parser.add_argument('--train_batch_size', type=int, default=32, help="batch size for client training")
parser.add_argument('--test_batch_size', type=int, default=32, help="batch size for testing data")

parser.add_argument('--model', type=str, default="MLP", help="network structure to be used for training", choices=['LR', 'MLP', 'CNN'])
parser.add_argument('--device', type=str, default="cpu", help="device for Torch", choices=['cpu', 'gpu'])
parser.add_argument('--frac_clients', type=float, default=0.1, help="proportion of clients to use for local updates")
parser.add_argument('--global_optimizer', type=str, default='fedavg', help="global optimizer to be used", choices=['fedavg', 'fedavgm', 'scaffold', 'fedadam', 'fedyogi'])
parser.add_argument('--global_epochs', type=int, default=100, help="number of global federated rounds")
parser.add_argument('--global_lr', type=float, default=1, help="learning rate for global steps")
parser.add_argument('--local_optimizer', type=str, default='sgd', help="local optimizer to be used", choices=['sgd', 'adam', 'pgd', 'scaffold'])
parser.add_argument('--local_epochs', type=int, default=20, help="number of local client training steps")
parser.add_argument('--local_lr', type=float, default=1e-4, help="learning rate for local updates")
parser.add_argument('--momentum', type=float, default=0.5, help="momentum value for SGD")
parser.add_argument('--mu', type=float, default=0.1, help="proximal coefficient for FedProx")
parser.add_argument('--beta1', type=float, default=0.9, help="parameter for FedAvgM and FedAdam")
parser.add_argument('--beta2', type=float, default=0.999, help="parameter for FedAdam")
parser.add_argument('--eps', type=float, default=1e-4, help="epsilon for adaptive methods")
parser.add_argument('--frac_byz_clients', type=float, default=0.0, help="proportion of clients that are picked in a round")
parser.add_argument('--is_attack', type=int, default=0, help="whether to attack or not")
parser.add_argument('--attack_type', type=str, default='label_flip', help="attack to be used", choices=['fall', 'label_flip', 'little', 'gaussian'])
parser.add_argument('--fall_eps', type=float, default=-5.0, help="epsilon value to be used for the Fall Attack")
parser.add_argument('--little_std', type=float, default=1.5, help="standard deviation to be used for the Little Attack")
parser.add_argument('--is_defense', type=int, default=0, help="whether to defend or not")
parser.add_argument('--defense_type', type=str, default='median', help="aggregation to be used", choices=['median', 'krum', 'trimmed_mean'])
parser.add_argument('--trim_ratio', type=float, default=0.1, help="proportion of updates to trim for trimmed mean")
parser.add_argument('--multi_krum', type=int, default=5, help="number of clients to pick after krumming")
parser.add_argument('--users_per_group',type=int,default=1,help='number of clients in one secure averaging round')
parser.add_argument('--global_momentum_param',type=float,default=1,help='the momentum to weight present iteration weights')

parser.add_argument('--batch_print_frequency', type=int, default=100, help="frequency after which batch results need to be printed to the console")
parser.add_argument('--global_print_frequency', type=int, default=1, help="frequency after which global results need to be printed to the console")
parser.add_argument('--global_store_frequency', type=int, default=100, help="frequency after which global results should be written to CSV")
parser.add_argument('--threshold_test_metric', type=float, default=0.9, help="threshold after which the code should end")

parameters = parser.parse_args()

with open('config.json') as f:
	json_vars = json.load(f)

parameters = vars(parameters)
parameters.update(json_vars)

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