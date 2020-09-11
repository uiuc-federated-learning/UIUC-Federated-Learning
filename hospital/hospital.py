from concurrent import futures
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
from src.local_train import LocalUpdate
from src.attacks import attack_updates
from src.defense import defend_updates

from collections import OrderedDict, Counter

from random import randint
import pickle
import time
import os

import urllib.parse
import urllib.request
import warnings
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
print(parameters)

np.random.seed(parameters['seed'])
torch.manual_seed(parameters['seed'])

############################### Loading Dataset ###############################
if parameters['data_source'] == 'MNIST':
	data_dir = 'data/'
	transformation = transforms.Compose([
		transforms.ToTensor(), 
		transforms.Normalize((0.1307,), (0.3081,))
	])
	train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transformation)
	test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transformation)
	print("Train and Test Sizes for %s - (%d, %d)"%(parameters['data_source'], len(train_dataset), len(test_dataset)))
	
################################ Sampling Data ################################
if parameters['sampling'] == 'iid':
	user_groups = iid(train_dataset, parameters['num_users'], parameters['seed'])
else:
	user_groups = non_iid(train_dataset, parameters['num_users'], parameters['num_shards_user'], parameters['seed'])

################################ Defining Model ################################
if parameters['model'] == 'LR':
	global_model = LR(dim_in=28*28, dim_out=10, seed=parameters['seed'])
elif parameters['model'] == 'MLP':
	global_model = MLP(dim_in=28*28, dim_hidden=200, dim_out=10, seed=parameters['seed'])
elif parameters['model'] == 'CNN' and parameters['data_source'] == 'MNIST':
	global_model = CNNMnist(parameters['seed'])
else:
	raise ValueError('Check the model and data source provided in the arguments.')

print("Number of parameters in %s - %d."%(parameters['model'], network_parameters(global_model)))

global_model.to(parameters['device'])
global_model.train()

global_weights = global_model.state_dict() # Setting the initial global weights
alpha=parameters['global_momentum_param']
############################ Initializing Placeholder ############################

# Momentum parameter 'v' for FedAvgM & `m` for FedAdam & FedYogi
# Control variates for SCAFFOLD (Last one corresponds to the server variate)
v = OrderedDict()
m = OrderedDict()
c = [OrderedDict() for i in range(len(user_groups) + 1)]

for k in global_weights.keys():
	v[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
	m[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
	for idx, i in enumerate(c):
		c[idx][k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)

################################ Defining Model ################################

#with open('results/%s_input.json'%(parameters['exp_name']), 'w') as f:
#	json.dump(parameters, f, indent=4)

train_loss_updated = []
train_loss_all = []
test_loss = []
train_accuracy = []
test_accuracy = []
mus = [parameters['mu'] for i in range(parameters['num_users'])]

num_classes = 10 # MNIST

idx = 0



class Federated(federated_pb2_grpc.FederatedServicer):

    def GetUpdatedModel(self, request, context):
        print("Sending model")
        ################################# Client Sampling & Local Training #################################
        global_model = pickle.loads(request.global_model)
        global_model.train()
        
        epoch = 0
        
        np.random.seed(epoch) # Picking a fraction of users to choose for training
        idxs_users = np.random.choice(range(parameters['num_users']), max(int(parameters['frac_clients']*parameters['num_users']), 1), replace=False)
        #Should change the below part later to accomodate sampling from selected users
        if parameters['is_attack'] == 1:
            idxs_byz_users = np.random.choice(idxs_users, max(int(parameters['frac_byz_clients']*len(idxs_users)), 1), replace=False)
        
        local_losses, local_sizes, control_updates = [], [], []


        if parameters['is_attack'] == 1 and parameters['attack_type'] == 'label_flip' and idx in idxs_byz_users:
            local_model = LocalUpdate(train_dataset, user_groups[idx], parameters['device'], 
                    parameters['train_test_split'], parameters['train_batch_size'], parameters['test_batch_size'], parameters['attack_type'], num_classes)
        else:
            local_model = LocalUpdate(train_dataset, user_groups[idx], parameters['device'], 
                    parameters['train_test_split'], parameters['train_batch_size'], parameters['test_batch_size'])

        w, c_update, c_new, loss, local_size = local_model.local_opt(parameters['local_optimizer'], parameters['local_lr'], 
                                                parameters['local_epochs'], global_model, parameters['momentum'], mus[idx], c[idx], c[-1], 
                                                epoch+1, idx+1, parameters['batch_print_frequency'])

        c[idx] = c_new
        
        # values = {'weights' : pickle.dumps(w)}

        control_updates.append(c_update)
        local_losses.append(loss)
        local_sizes.append(local_size)
        #print(idx, np.unique(np.array([train_dataset.targets.numpy()[i] for i in user_groups[idx]])))

        train_loss_updated.append(sum(local_losses)/len(local_losses)) # Appending global training loss
        return federated_pb2.ModelWeights(weights=pickle.dumps(w), local_size=local_size)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederatedServicer_to_server(Federated(), server)
    server.add_insecure_port('[::]:1234')
    server.start()
    server.wait_for_termination()

logging.basicConfig()
serve()