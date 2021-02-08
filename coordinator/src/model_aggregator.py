import numpy as np

import torch
from torch import nn
from time import time

# from torchvision import datasets, transforms

from .models import LR, MLP, CNNMnist
from .utils import global_aggregate, network_parameters, test_inference
import torchvision

import copy
import itertools

from collections import OrderedDict

class ModelAggregator():
    local_weights = []
    local_sizes = []

    c = [OrderedDict() for i in range(2)]

    epoch = 0

    global_model = None
    global_weights = None

    def __init__(self, p):
        self.parameters = p

        np.random.seed(self.parameters['seed'])
        torch.manual_seed(self.parameters['seed'])

        ################################ Defining Model ################################
        if self.parameters['model'] == 'LR':
            self.global_model = LR(dim_in=28*28, dim_out=10, seed=self.parameters['seed'])
        elif self.parameters['model'] == 'MLP':
            self.global_model = MLP(dim_in=28*28, dim_hidden=200, dim_out=10, seed=self.parameters['seed'])
            self.example_input = torch.ones((self.parameters['train_batch_size'],1,28,28))
        elif self.parameters['model'] == 'CNN' and self.parameters['data_source'] == 'MNIST':
            self.global_model = CNNMnist(self.parameters['seed'])
        elif self.parameters['model'] == 'DENSENET' and self.parameters['data_source'] == 'COVID':
            self.global_model = torchvision.models.densenet121(pretrained=True)
            # Fine tune
            for param in self.global_model.parameters():
                param.requires_grad = False if self.parameters['finetune'] else True
            num_ftrs = self.global_model.classifier.in_features
            self.example_input = torch.ones((self.parameters['train_batch_size'],3,224,224))
            self.global_model.classifier = nn.Linear(num_ftrs, 2)
        elif self.parameters['model'] == 'RESNET' and self.parameters['data_source'] == 'COVID':
            self.global_model = torchvision.models.resnet101(pretrained=True)
            # Fine tune
            print(self.global_model.parameters())
            for param in self.global_model.parameters():
                param.requires_grad = False if self.parameters['finetune'] else True
            num_ftrs = self.global_model.fc.in_features
            self.example_input = torch.ones((self.parameters['train_batch_size'],3,224,224))
            self.global_model.fc = nn.Linear(num_ftrs, 2)
        else:
            raise ValueError('Check the model and data source provided in the arguments.')

        print("Number of parameters in %s - %d."%(self.parameters['model'], network_parameters(self.global_model)))

        self.global_model.to(self.parameters['device'])

        self.global_weights = self.global_model.state_dict() # Setting the initial global weights

        ############################ Initializing Placeholder ############################

        for k in self.global_weights.keys():
            #v[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
            #m[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
            for idx, i in enumerate(self.c):
                self.c[idx][k] = torch.zeros(self.global_weights[k].shape, dtype=self.global_weights[k].dtype)





    def add_hospital_data(self, weights, local_size):
        self.local_weights.append(weights)
        self.local_sizes.append(local_size)
    
    def aggregate(self):
        gw = copy.deepcopy(self.global_weights)
        
        self.global_model.load_state_dict(gw)

        self.global_weights = global_aggregate(self.parameters['global_optimizer'], self.global_weights, self.local_weights, self.local_sizes,
                                            self.parameters['global_momentum_param'], self.parameters['global_lr'], self.parameters['beta1'], self.parameters['beta2'],
                                            self.parameters['eps'], self.epoch+1)

        start = time()
        # self.interpret_weights(self.global_weights, self.parameters['shift_amount'])
        end = time()
        print(f'Shifting weights took {end-start} seconds')

        self.epoch += 1
        self.global_model.load_state_dict(self.global_weights)

        self.local_weights = []
        self.local_sizes = []
        
        return self.global_model
