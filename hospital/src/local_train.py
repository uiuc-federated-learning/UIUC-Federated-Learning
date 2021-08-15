import copy
from collections import OrderedDict

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.optimizers import StochasticControl
from src.utils import DatasetSplit

def iid(dataset, num_users, seed):
	"""
	Divides the given dataset in a IID fashion into specified number of users.

	Args:
		dataset (tensor) : dataset to be partitioned
		num_users (int) : # users to be created
		seed (int) : Random seed value
	"""
	np.random.seed(seed)
	
	num_items = int(len(dataset) / num_users)
	rem_items = len(dataset) % num_users
	if rem_items == 0:
		print("Each user will get %d samples from the training set."%(num_items))
	else:
		print("Each user will get %d samples from the training set. %d samples are discarded."%(num_items, rem_items))

	user_groups = {} 
	all_idxs = list(range(len(dataset)))
	
	for i in range(num_users):
		user_groups[i] = list(np.random.choice(all_idxs, num_items, replace=False))
		all_idxs = list(set(all_idxs) - set(user_groups[i]))
	
	return user_groups


def create_federated_dataloader(image_dataset, batch_size, num_workers, num_clients, client_number, seed=10082020):
    client_groups = iid(image_dataset, num_clients, seed)
    subsets = [torch.utils.data.Subset(image_dataset, client_groups[i]) for i in range(num_clients)]
    dataloader = torch.utils.data.DataLoader(subsets[client_number], batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
    return dataloader

class LocalUpdate(object):
	
	def __init__(self, train_dataset, test_dataset, client_num, num_clients, device, train_test_split=0.8,
				train_batch_size=32, test_batch_size=32, attack=None,
				num_classes=None):
		"""
		Args:
			dataset (tensor) : Global data
			idxs (list) : List of indexes corresponding to the global data for making it local
			device (str) : One from ['cpu', 'cuda'].
			train_test_split (float) : Proportion of client data to be split for training and testing
			train_batch_size (int) : Batch size of the training samples
			test_batch_size (int) : Batch size of the testing samples
			attack (str) : Type of attack, one of ['label_flip']
			num_classes (int) : Total number of classes for label_flip attack
		"""
	
		self.device = device
		self.train_test_split = train_test_split
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.attack = attack
		self.num_classes = num_classes
		# self.criterion = nn.NLLLoss().to(self.device) # Default criterion set to NLL loss function
		self.criterion = nn.CrossEntropyLoss().to(self.device)

		# train_idxs = KFold(n_splits=num_clients, random_state=1, shuffle=True).split(train_dataset)[client_num]

		self.train_loader = create_federated_dataloader(train_dataset, self.train_batch_size, 0, num_clients, client_num)
		self.test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size)

	def local_opt(self, optimizer, lr, epochs, global_model, modelbuffer, momentum=0, mu=0.01, client_controls=[], 
				server_controls=[], global_round=0, client_no=0, batch_print_frequency=100):
		"""
		Local client optimization in the form of updates/steps.

		Args:
			optimizer (str) : Local optimizer to be used for training
			lr (float) : step-size for client training
			epochs (int) : # local steps to be taken
			global_model (model state) : Initial global model
			momentum (float) : Momentum parameter for SGD
			mu (float) : Coefficient for the proximal term in FedProx
			client_controls (list) : Control variates for the client
			server_controls (list) : Control variates for the server
			global_round (int) : Current global federated round number
			client_no (int) : Current client number
			batch_print_frequency (int) : Epoch cut-off for printing batch-level measures
		"""
		
		global_params = [global_model.state_dict()[key] for key in global_model.state_dict()]
		server_controls_list = [server_controls[key] for key in server_controls.keys()]
		client_controls_list = [client_controls[key] for key in client_controls.keys()]
		
		# If using JIT, need to copy the buffer and load the model since we can't deepcopy a jitted model.
		if modelbuffer != None:
			local_model = torch.jit.load(modelbuffer)
		else:
			local_model = copy.deepcopy(global_model)
		# Set model to ``train`` mode
		local_model.train()

		# Set local optimizer
		if optimizer == 'sgd':
			opt = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)
		elif optimizer == 'adam':
			opt = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
		elif optimizer in ['fedprox', 'scaffold']:
			opt = StochasticControl(local_model.parameters(), lr=lr, mu=mu)
		else:
			raise ValueError("Please specify a valid value for the optimizer from ['sgd', 'adam', 'fedprox', 'scaffold'].")

		epoch_loss = []
							 
		for epoch in range(epochs):
			print("\nStarting local epoch",epoch+1)
			batch_loss = []
			
			for batch_idx, (images, labels) in enumerate(self.train_loader):
				images, labels = images.to(self.device), labels.to(self.device)
				local_model.zero_grad()
				log_probs = local_model(images)
				loss = self.criterion(log_probs, labels)
				loss.backward()
				if optimizer in ['fedprox', 'scaffold']:
					opt.step(optimizer, global_params, server_controls_list, client_controls_list)
				else:
					opt.step()

				if (batch_idx+1) % batch_print_frequency == 0:
					msg = '| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
					print(msg.format(global_round, epoch, batch_idx * len(images), len(self.train_loader.dataset),
						100. * batch_idx / len(self.train_loader), loss.item()))

				batch_loss.append(loss.item())

			epoch_loss.append(sum(batch_loss)/len(batch_loss))

		# Finding the local update in parameters
		local_changes = OrderedDict()
		for k in global_model.state_dict():
			local_changes[k] = local_model.state_dict()[k]

		# Updating local client variates
		control_changes = OrderedDict()
		if optimizer == 'scaffold':
			for key in client_controls.keys():
				control_changes[key] = torch.mul(-1., client_controls[key])
				client_controls[key] = client_controls[key] - server_controls[key] - local_changes[key]# torch.div(local_params[key], (lr*local_epochs))
				control_changes[key] += client_controls[key]

		return local_changes, control_changes, client_controls, epoch_loss[-1], len(self.train_loader)

	def inference(self, global_model):
		"""
		Returns the inference accuracy and loss.

		Args:
			global_model (model state) : Global model for evaluation
		"""
		
		global_model.eval()
		loss, total, correct = 0.0, 0.0, 0.0

		for batch_idx, (images, labels) in enumerate(self.test_loader):

			images, labels = images.to(self.device), labels.to(self.device)

			outputs = global_model(images)
			batch_loss = self.criterion(outputs, labels)
			loss += batch_loss.item()

			_, pred_labels = torch.max(outputs, 1)
			pred_labels = pred_labels.view(-1)
			correct += torch.sum(torch.eq(pred_labels, labels)).item()
			total += len(labels)

		return correct/total, loss