import copy
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
	
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

		images_1, labels_1 = images.to(device), labels.to(device)

		outputs = global_model(images_1)
		batch_loss = criterion(outputs, labels_1)
		loss += batch_loss.item()

		# Prediction
		_, pred_labels = torch.max(outputs, 1)
		pred_labels = pred_labels.view(-1)
		correct += torch.sum(torch.eq(pred_labels, labels)).item()
		total += len(labels)
	
	return correct/total, loss/total