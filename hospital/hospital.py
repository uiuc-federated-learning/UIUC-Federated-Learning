from concurrent import futures
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
from src.local_train import LocalUpdate
from src.flag_parser import Parser

from collections import OrderedDict, Counter

from random import randint
import pickle
import time
import os

import urllib.parse
import urllib.request
import warnings
warnings.filterwarnings("ignore")

parser = Parser()
parameters = parser.parse_arguments()

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

class Hospital(federated_pb2_grpc.HospitalServicer):

    def ComputeUpdatedModel(self, global_model, context):
        print("Sending model")
        ################################# Client Sampling & Local Training #################################
        global_model = pickle.loads(global_model.weights)
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
        accuracy(global_model, epoch)
        return federated_pb2.TrainedModel(model=federated_pb2.Model(weights=pickle.dumps(w)), training_samples=local_size) 

def accuracy(global_model, epoch):
    test_acc, test_loss_value = test_inference(global_model, test_dataset, parameters['device'], parameters['test_batch_size'])
    test_accuracy.append(test_acc)
    test_loss.append(test_loss_value)

    if (epoch+1) % parameters['global_print_frequency'] == 0 or (epoch+1) == parameters['global_epochs']:
        msg = '| Global Round : {0:>4} | TeLoss - {1:>6.4f}, TeAcc - {2:>6.2f} %, TrLoss (U) - {3:>6.4f}'

        if parameters['train_test_split'] != 1.0:
            msg = 'TrLoss (A) - {4:>6.4f} % , TrAcc - {5:>6.2f} %'
            print(msg.format(epoch+1, test_loss[-1], test_accuracy[-1]*100.0, train_loss_updated[-1], 
                            train_loss_all[-1], train_accuracy[-1]*100.0))
        else:
            print(msg.format(epoch+1, test_loss[-1], test_accuracy[-1]*100.0, train_loss_updated[-1]))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_HospitalServicer_to_server(Hospital(), server)
    port = parameters['port']
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    server.wait_for_termination()

logging.basicConfig()
serve()
