from concurrent import futures
import logging

import numpy as np
import copy
import os

import torch
from torchvision import datasets, transforms

from .utils import network_parameters, test_inference
from .models import *
from .local_train import LocalUpdate
from .flag_parser import Parser

from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from random import randint

import pickle

import warnings
warnings.filterwarnings("ignore")

class ModelTraining():
    def get_data(self):
        data_dir = 'data/'
        if self.parameters['data_source'] == 'MNIST':
            transformation = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transformation)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transformation)
            self.parameters['batch_print_frequency'] = 500
        elif self.parameters['data_source'] == 'COVID':
            transformation = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transformation)
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transformation)

            if len(train_dataset) == 0 or len(test_dataset) == 0:
                raise ValueError("Dataset is empty.")
        else:
            raise ValueError("Not a valid data_source type.")

        self.parameters['batch_print_frequency'] = (len(train_dataset) //2) // 3

        print("Train and Test Sizes for %s - (%d, %d)"%(self.parameters['data_source'], len(train_dataset), len(test_dataset)))
        return train_dataset, test_dataset

    def __init__(self, parameters):
        self.parameters = parameters
        np.random.seed(parameters['seed'])
        torch.manual_seed(parameters['seed'])

        ############################### Loading Dataset ###############################
        self.train_dataset, self.test_dataset = self.get_data()
            
        ################################ Sampling Data ################################
        self.user_groups = {}
        self.user_groups[0] = list(range(len(self.test_dataset)))# list(np.random.choice(list(range(len(dataset))), num_items, replace=False))

        self.alpha = self.parameters['global_momentum_param']

        self.train_loss_updated = []
        self.train_loss_all = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.mus = self.parameters['mu']

    def setVars(self, global_weights):
        ############################ Initializing Placeholder ############################

        # Momentum parameter 'v' for FedAvgM & `m` for FedAdam & FedYogi
        # Control variates for SCAFFOLD (Last one corresponds to the server variate)
        self.v = OrderedDict()
        self.m = OrderedDict()
        self.c = OrderedDict()

        for k in global_weights.keys():
            self.v[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
            self.m[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
            self.c[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)


    def accuracy(self, global_model, epoch):
        self.test_acc, self.test_loss_value = test_inference(global_model, self.test_dataset, self.parameters['device'], self.parameters['test_batch_size'])
        self.test_accuracy.append(self.test_acc)
        self.test_loss.append(self.test_loss_value)

        if (self.epoch+1) % self.parameters['global_print_frequency'] == 0 or (self.epoch+1) == self.parameters['global_epochs']:
            if self.parameters['train_test_split'] != 1.0:
                msg = 'TrLoss (A) - {4:>6.4f} % , TrAcc - {5:>6.2f} %'
                print(msg.format(self.epoch+1, self.test_loss[-1], self.test_accuracy[-1]*100.0, self.train_loss_updated[-1], 
                                self.train_loss_all[-1], self.train_accuracy[-1]*100.0))
            elif len(self.train_loss_updated) == 0:
                msg = '| Global Round : {0:>4} | TeLoss - {1:>6.4f}, TeAcc - {2:>6.2f} %'
                print(msg.format(self.epoch+1, self.test_loss[-1], self.test_accuracy[-1]*100.0))
            else:
                msg = '| Global Round : {0:>4} | TeLoss - {1:>6.4f}, TeAcc - {2:>6.2f} %, TrLoss (U) - {3:>6.4f}'
                print(msg.format(self.epoch+1, self.test_loss[-1], self.test_accuracy[-1]*100.0, self.train_loss_updated[-1]))
    
    def ComputeUpdatedModel(self, model_obj, modelbuffer):
        ################################# Client Sampling & Local Training #################################
        global_model = model_obj
        global_model.train()
        
        self.setVars(global_model.state_dict())

        self.epoch = 0
        print("=> Initial Accuracy:")
        self.accuracy(global_model, self.epoch)
        
        np.random.seed(randint(1,777)) # Picking a fraction of users to choose for training
        idxs_users = np.random.choice(range(self.parameters['num_users']), max(int(self.parameters['frac_clients']*self.parameters['num_users']), 1), replace=False)
    
        local_losses, local_sizes = [], []


        
        local_model = LocalUpdate(self.train_dataset, self.test_dataset, self.parameters['client_num'], self.parameters['num_clients'], self.parameters['device'], 
                self.parameters['train_test_split'], self.parameters['train_batch_size'], self.parameters['test_batch_size'])

        
        
        w, c_update, c_new, loss, local_size = local_model.local_opt(self.parameters['local_optimizer'], self.parameters['local_lr'], 
                                                self.parameters['local_epochs'], global_model, modelbuffer, self.parameters['momentum'], self.mus, self.c, self.c, 
                                                self.epoch+1, 1, self.parameters['batch_print_frequency'])

        global_model.load_state_dict(w)

        self.c = c_new
        
        local_losses.append(loss)
        local_sizes.append(local_size)
        
        self.train_loss_updated.append(sum(local_losses)/len(local_losses)) # Appending global training loss
        self.accuracy(global_model, self.epoch)
        
        return w , local_size