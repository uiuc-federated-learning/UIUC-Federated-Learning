import copy
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

BITS = 64

def interpret_weights(state_dict, new_dict, shift_amount):

	#for key, value in state_dict.items():
		# print("TYPE", type(state_dict[key][0][0][0][0]), state_dict[key].shape)
		# if (key == "features.norm0.weight"): print("Preshift weights (no mod):", state_dict[key][:10])
		# state_dict[key] = torch.fmod(state_dict[key], 2**BITS)
		#z_positive = torch.mul(state_dict[key].le(2**(BITS-1)), state_dict[key])
		#z_negative = torch.mul(state_dict[key].ge(2**(BITS-1)), state_dict[key] - 2**BITS)

		#state_dict[key] = z_positive + z_negative
	power = (1<<shift_amount)

	for key, value in state_dict.items():
		new_dict[key] = (state_dict[key].double() / power).float()
	# return new_dict


def global_aggregate(global_optimizer, global_weights, local_updates, local_sizes, alpha,
					global_lr=1., beta1=0.9, beta2=0.999, eps=1e-4, step=None, shift_amount=24):
	"""
	Aggregates the local client updates to find a focused global update.

	Args:
		global_optimizer (str) : Optimizer to be used for the steps
		global_weights (OrderedDict) : Initial state of the global model (which needs to be updated here)
		local_updates (list of OrderedDict) : Contains the update differences (delta) between global and local models
		global_lr (float) : Stepsize for global step
		beta1 (float) : Role of ``beta`` in FedAvgM, otheriwse analogous to beta_1 and beta_2 famous in literature for Adaptive methods
		beta2 (float) : Same as above
		step (int) : Current epoch number to configure ADAM and YOGI properly
	"""
	
	total_size = sum(local_sizes)

	################################ FedAvg ################################
	# Good illustration provided in SCAFFOLD paper - Equations (1). (https://arxiv.org/pdf/1910.06378.pdf)
	if global_optimizer == 'fedavg':
		w = copy.deepcopy(global_weights).double()
		temp_copy=copy.deepcopy(global_weights)

		# Add up weights
		for key in w.keys():
			for i in range(len(local_updates)):
				w[key] += torch.div(local_updates[i][key], len(local_sizes))
		
		# Interpret Weights
		interpret_weights(w, shift_amount) # HARD CODE

		for key in w.keys():
			w[key] = (1-alpha)*temp_copy[key]+alpha*w[key]
		return w
	elif global_optimizer == 'simpleavg':
		# TODO: remove deepcopy
		w = copy.deepcopy(global_weights)
		w1 = copy.deepcopy(global_weights)

		for key in w.keys():
			w[key] = torch.zeros(w[key].shape, dtype=torch.long)
			w1[key] = torch.zeros(w[key].shape, dtype=torch.float)
			for i in range(len(local_updates)):
				w[key] += local_updates[i][key]

		# TODO: Hard Code
		interpret_weights(w, w1, 24)

		for key in w1.keys():
			w1[key] = torch.div(w1[key], len(local_sizes))
			
		return w1
	else:
		raise ValueError('Check the global optimizer for a valid value.')
	
def network_parameters(model):
	"""
	Calculates the number of parameters in the model.

	Args:
		model : PyTorch model used after intial weight initialization
	"""
	total_params = 0
	
	for param in list(model.parameters()):
		curr_params = 1
		for p in list(param.size()):
			curr_params *= p
		total_params += curr_params
		
	return total_params

class DatasetSplit(Dataset):
	"""
	An abstract dataset class wrapped around Pytorch Dataset class.
	"""

	def __init__(self, dataset, idxs):

		self.dataset = dataset
		self.idxs = [int(i) for i in idxs]

	def __len__(self):

		return len(self.idxs)

	def __getitem__(self, item):
		
		image, label = self.dataset[self.idxs[item]]

		return torch.tensor(image), torch.tensor(label)

def test_inference(global_model, test_dataset, device, test_batch_size=128):
	"""
	Evaluates the performance of the global model on hold-out dataset.

	Args:
		global_model (model state) : Global model for evaluation
		test_dataset (tensor) : Hold-out data available at the server
		device (str) : One from ['cpu', 'cuda'].
		test_batch_size (int) : Batch size of the testing samples
	"""

	test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
	criterion = nn.NLLLoss().to(device)
	global_model.eval()

	loss, total, correct = 0.0, 0.0, 0.0

	for batch_idx, (images, labels) in enumerate(test_loader):

		images, labels = images.to(device), labels.to(device)

		outputs = global_model(images)
		batch_loss = criterion(outputs, labels)
		loss += batch_loss.item()

		# Prediction
		_, pred_labels = torch.max(outputs, 1)
		pred_labels = pred_labels.view(-1)
		correct += torch.sum(torch.eq(pred_labels, labels)).item()
		total += len(labels)
	
	return correct/total, loss/total