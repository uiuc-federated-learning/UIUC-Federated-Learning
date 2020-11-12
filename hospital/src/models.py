import torch
import torch.nn.functional as F
from torch import nn

	
class MLP(nn.Module):
	pass
	"""
	Multi Layer Perceptron with a single hidden layer.
	"""
	
	def __init__(self, dim_in, dim_hidden, dim_out, seed):
		"""
		Args:
			dim_in (int) : Input dimension
			dim_hidden (int) : # units in the hidden layer
			dim_out (int) : Output dimension
			seed (int) : Random seed value
		"""
		
		super(MLP, self).__init__()
		
		pass

	def forward(self, x):
		
		x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
		x = self.input(x)
		x = self.relu(x)
		x = self.layer_hidden(x)
		x = self.relu(x)
		x = self.output(x)

		return F.log_softmax(x, dim=1)
