from math import floor
from copy import deepcopy

import torch
import torch.nn as nn


TOPK_INIT_METHODS = {
    'xavier_normal': nn.init.xavier_normal_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'orthogonal': nn.init.orthogonal_,
    'normal': nn.init.normal_,
    'uniform': nn.init.uniform_,
    'eye': nn.init.eye_,
}


class TopKLinear(nn.Module):
    """
    A linear layer that retains only the top K strongest synaptic weights for
    in_features.

    This module implements a linear transformation with weights constrained to 
    be positive. For each output neuron, only the top K weights are kept 
    , and the rest are set to zero. This simulates a neuron receiving 
    inputs only from its strongest synaptic connections.

    Parameters
    ----------

    in_features : int
        The number of input features.
    
    out_features : int
        The number of output features.

    K : int
        The number of strongest synapses to keep per dendritic branch.

    param_space : str, optional
        The parameter space for the weights. Options are 'log' and 'presigmoid'.
        If `'log'`, the weights are parameterized as exponentials of `pre_w`.
        If `'presigmoid'`, the weights are parameterized as sigmoids of `pre_w`.
        Defaults to 'log'.
    
    Attributes
    ----------

    pre_w : torch.nn.Parameter
        The raw weights before applying the exponential or sigmoid 
        transformation. Initialized with small negative values to ensure 
        positive weights after transformation.

    K : int
        The number of strongest synapses to keep per dendritic branch.

    param_space : str
        The parameter space for the weights.

    Methods
    -------

    forward(x)
        Performs a forward pass through the layer.

    weight()
        Returns the transformed synaptic weights after applying the exponential 
        or sigmoid.
    
    weight_mask()
        Returns a mask tensor indicating the top K synaptic connections per 
        output neuron.

    pruned_weight()
        Returns the pruned synaptic weights after applying the mask.

    weighted_synapses(cell_weights, prune=False)
        Returns the weighted synapses for a given set of cell

    Notes
    -----
    - All weights are constrained to be positive.
    - The prunning is done dynamically during the forward pass.

    Examples
    --------

    >>> import torch
    >>> from torch import nn
    >>> topk_linear = TopKLinear(in_features=10, out_features=5, K=3)
    >>> x = torch.randn(4, 10)  # Batch of 4 samples
    >>> output = topk_linear(x)
    >>> print(output.shape)
    torch.Size([4, 5])
    """
    def __init__(
            self, 
            in_features, 
            out_features, 
            K, 
            param_space = 'log',
            init_method = 'xavier_normal',
            init_params = None,
        ):
        super(TopKLinear, self).__init__()

        # Co-release sign matrix
        self.corelease_sign = nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad=True)
        self.corelease_sign.data.uniform_(-1, 1)  # Initialize with random signs
        self.corelease_sign.data = self.corelease_sign.sign()  # Ensure values are -1 or 1

        # LHb to DAN sign matrix
        self.negative_sign = nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad=False)
        self.negative_sign.data.uniform_(-1, -1)  # Initialize with random signs
        self.negative_sign.data = self.negative_sign.sign()  # Ensure values are -1 or 1

        # make all weights positive
        self.pre_w = nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad = True)

        if not isinstance(K, int):
            raise TypeError('K must be an integer')
        
        if K < 1:
            raise ValueError('K must be greater than or equal to 1')
        
        if K > in_features:
            raise ValueError(
                f'K must be less than or equal to the number of input ',
                f'features. (K = {K}, in_features = {in_features})')

        self.K = K
        self.param_space = param_space
        self.init_method = init_method
            
    def initialize(self):
        if self.init_method in TOPK_INIT_METHODS:
            init_func = TOPK_INIT_METHODS[self.init_method]
            init_func(self.pre_w)
            #TODO: Accept hyperparameters for initialization
            # init_func(self.pre_w, **self.init_params) 
        else:
            raise ValueError(
                f'Invalid initialization method: {self.init_method}. ',
                f'Choose from {list(TOPK_INIT_METHODS.keys())}')
    
    def decay_weights(self, weight_decay = 0.1):
        with torch.no_grad():
            self.pre_w.data -= (weight_decay * self.weight_mask())
        
    def forward(self, x):
        # identify top K strongest synaptic connections onto each dendritic branch
        pruned_weight = self.pruned_weight()
        # matrix multiply inputs and synaptic weights
        return torch.mm(x, pruned_weight.t())
    
    def weight(self):
        if self.param_space == 'log':
            return self.pre_w.exp() * self.sign_matrix
        elif self.param_space == 'presigmoid':
            return torch.sigmoid(self.pre_w) * self.sign_matrix
    
    def log_weight(self):
        return self.weight().log()

    def weight_mask(self):
        topK_indices = torch.topk(self.pre_w, 
                                  self.K, 
                                  dim = -1, 
                                  largest = True, 
                                  sorted = False)[1]
        # initialize and populate masking matrix
        mask = torch.zeros_like(
            self.pre_w, 
            device = self.pre_w.device, 
            dtype = self.pre_w.dtype,
            )
        mask[torch.arange(self.pre_w.shape[0])[:,None], topK_indices] = 1
        return mask

    def pruned_weight(self):
        return self.weight_mask() * self.weight()
    
    def log_pruned_weight(self):
        return ((self.weight_mask() - 1) * 10) + self.log_weight()
    
    def weighted_synapses(self, cell_weights, prune = False):
        if prune:
            synapse_weights = self.pruned_weight()
        else:
            synapse_weights = self.weight()
        
        weighted_synapses = cell_weights[:,None] * synapse_weights
        return weighted_synapses.sum(dim = 0 )









