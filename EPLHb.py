import torch
import torch.nn as nn
import math


class EPLHb(nn.Module):
  def __init__(self, EP_size, LHb_size, DAN_size, 
               LHb_rnn: bool=False,
               fixed_sign: bool=False, real_circuit: bool=True, 
               prob_EP_to_LHb: float=1, prob_LHb_to_LHb: float=1, prob_LHb_to_DAN: float=1):
    super(EPLHb,self).__init__()
    
    # Initialize layers
    if LHb_rnn: 
      self.LHb_RNN = nn.RNN(EP_size, LHb_size, batch_first=True, bias=True)
    else: 
      self.EP_to_LHb = nn.Linear(EP_size, LHb_size, bias=True)
      # nn.init.xavier_normal_(self.EP_to_LHb.weight)

    self.LHb_to_DAN = nn.Linear(LHb_size, DAN_size, bias=True)
    # nn.init.xavier_normal_(self.LHb_to_DAN.weight)
   
    with torch.no_grad():
      if LHb_rnn:
        # Make EP to LHb sparse
        n_zeros = int((1-prob_EP_to_LHb) * self.LHb_RNN.weight_ih_l0.numel())
        sparse_idx_EP = torch.randperm(self.LHb_RNN.weight_ih_l0.numel())[:n_zeros]
        self.LHb_RNN.weight_ih_l0.data.view(-1)[sparse_idx_EP] = 0
      else:
        # Make EP to LHb sparse
        n_zeros = int((1-prob_EP_to_LHb) * self.EP_to_LHb.weight.numel())
        sparse_idx_EP = torch.randperm(self.EP_to_LHb.weight.numel())[:n_zeros]
        self.EP_to_LHb.weight.data.view(-1)[sparse_idx_EP] = 0

      # Make LHb to DAN sparse
      n_zeros = int((1-prob_LHb_to_DAN) * self.LHb_to_DAN.weight.numel())
      sparse_idx_LHb = torch.randperm(self.LHb_to_DAN.weight.numel())[:n_zeros]
      self.LHb_to_DAN.weight.data.view(-1)[sparse_idx_LHb] = 0
      
      if real_circuit: 
        # Make LHb to LHb all excitatory
        if LHb_rnn: self.LHb_RNN.weight_hh_l0.data = torch.abs(self.LHb_RNN.weight_hh_l0.data)
        # Make LHb to DAN all negative
        self.LHb_to_DAN.weight.data = -torch.abs(self.LHb_to_DAN.weight)
      
      # Turn into fixed sign (obey Dale's law)
      pos_neurons = {}
      neg_neurons = {}
      if fixed_sign:
        for name, param in self.named_parameters():
          if "weight" in name:
            # Find each neuron is excitatory or inhibitory
            pos_neurons[name] = torch.sum(param.data, axis=0) >= 0
            neg_neurons[name] = torch.sum(param.data, axis=0) < 0
            # Make all weights of that neuron excitatory or inhibitory
            param.data[:,pos_neurons[name]] = torch.sign(param[:,pos_neurons[name]])*param[:,pos_neurons[name]]
            param.data[:,neg_neurons[name]] = -torch.sign(param[:,neg_neurons[name]])*param[:,neg_neurons[name]]
    
    self.relu = nn.ReLU()
    # self.tanh = nn.Tanh()
    self.tanh = nn.Tanh()

    # Store information
    self.EP_size = EP_size
    self.LHb_size = LHb_size
    self.DAN_size = DAN_size
    self.LHb_rnn = LHb_rnn
    self.real_circuit = real_circuit
    self.fixed_sign = fixed_sign
    self.sparse_idx_EP = sparse_idx_EP
    self.sparse_idx_LHb = sparse_idx_LHb
    self.pos_neurons = pos_neurons
    self.neg_neurons = neg_neurons
    self.init_weights = self.record_params(calc_sign=False)

  def enforce_weights(self):
    with torch.no_grad():
      # Make EP to LHb sparse
      if self.LHb_rnn: self.LHb_RNN.weight_ih_l0.data.view(-1)[self.sparse_idx_EP] = 0
      else: self.EP_to_LHb.weight.data.view(-1)[self.sparse_idx_EP] = 0

      # Keep LHb to DAN sparse
      self.LHb_to_DAN.weight.data.view(-1)[self.sparse_idx_LHb] = 0
      
      if self.real_circuit:
        # Make LHb to LHb all excitatory
        if self.LHb_rnn: self.LHb_RNN.weight_hh_l0.data = torch.max(self.LHb_RNN.weight_hh_l0.data, 0*self.LHb_RNN.weight_hh_l0.data)
        # Make LHb to DAN all negative
        self.LHb_to_DAN.weight.data=torch.minimum(self.LHb_to_DAN.weight, 0*self.LHb_to_DAN.weight)

  def forward(self, input):
    if self.LHb_rnn:
      # Initialize hidden state with zeros
      h0 = torch.zeros(1, input.size(0), self.LHb_size)
      hidden, _ = self.LHb_RNN(input, h0)
      LHB_out = hidden[:, -1, :]
    else: 
      LHB_act = self.EP_to_LHb(input)
      LHB_out = self.tanh(LHB_act)

    DAN_act = self.LHb_to_DAN(LHB_out)
    DAN_out = self.tanh(DAN_act)

    return DAN_out
  
  def record_params(self, calc_sign: bool=True):
    # Save the network weights
    recorded_params = {}
    for name, param in self.named_parameters():
        if param.requires_grad:
          with torch.no_grad():
            cur_data = param.data.detach().cpu().clone()
            recorded_params[name] = (cur_data)
          
          if calc_sign:
            print(name)
            frac_pos = 100*(torch.sum(cur_data > 0)/cur_data.numel()).numpy()
            frac_zero = 100*(torch.sum(cur_data == 0)/cur_data.numel()).numpy()
            frac_neg = 100*(torch.sum(cur_data < 0)/cur_data.numel()).numpy()
            print(' Positive: ' + str(frac_pos) + '%; Negative: ' + str(frac_neg) + '%; Zero: ' + str(frac_zero) + '%')
            
    return recorded_params
  
  def train_model(self,num_epochs,train_loader,optimizer,
                  loss: str='MSE',
                  print_epoch: bool=True,
                  test_loader: torch.utils.data.DataLoader=None):

    # Define loss function
    training_loss = []
    test_accuracy = []
    if loss == 'MSE': loss_function = nn.MSELoss()
    elif loss in 'CrossEntropyLoss': loss_function = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Move DataLoader to device
    
    # Train the network
    for epoch in range(num_epochs):
      for i, (data,labels) in enumerate(train_loader):
        optimizer.zero_grad()

        if self.LHb_rnn: data = data.view(-1,1, self.EP_size)
        elif data.ndim != 2: data = data.view(-1, self.EP_size)
        outputs = self(data)

        loss = loss_function(outputs.squeeze(), labels)
        training_loss.append(loss.data.cpu())

        loss.backward()
        optimizer.step(init_weights=list(self.init_weights.values()))
        self.enforce_weights()
        
        # Calculate Accuracy
        if i % 100 == 0:
          if test_loader is not None:
            self.eval()
            correct, total = 0, 0
            # Iterate through test dataset
            for test_data, test_labels in test_loader:
              if self.LHb_rnn: test_data = test_data.view(-1,1, self.EP_size)
              elif test_data.ndim != 2: test_data = test_data.view(-1, self.EP_size)

              test_outputs = self(test_data)
              _, predicted = torch.max(test_outputs.data, 1)
              total += test_labels.size(0)
              correct += (predicted == test_labels).sum()

            accuracy = 100 * correct / total
            test_accuracy.append(accuracy)
            if print_epoch:
              print('Epoch [%d/%d], Iteration: %d, Loss: %.4f, Accuracy: %.4f' %(epoch+1, num_epochs, i, loss.data, accuracy))
          
          else:
            if print_epoch: print('Epoch [%d/%d], Iteration: %d, Loss: %.4f'  %(epoch+1, num_epochs, i, loss.data))
      # scheduler.step()

    return training_loss, test_accuracy
  


class gd(torch.optim.Optimizer): 
  def __init__(self, params, lr=0.01, fixed_sign: bool = False): 
    defaults = dict(lr=lr, fixed_sign=fixed_sign) 
    super(gd, self).__init__(params, defaults) 

  def step(self, init_weights=None): 
    for group in self.param_groups: 
      for i, p in enumerate(group['params']): 
        if p.grad is None: continue
        p.data = p.data - group['lr']*p.grad.data

        if group['fixed_sign']:
          flip_mask = init_weights[i].sign()*p.data.sign()<0
          p.data[flip_mask] = 0



class adam(torch.optim.Optimizer):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, fixed_sign: bool = False): 
		defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fixed_sign=fixed_sign)
		super(adam, self).__init__(params, defaults) 

	def step(self, init_weights=None):
		for group in self.param_groups:
			for i, p in enumerate(group['params']): 
				if p.grad is None: continue
				grad = p.grad.data 
				if grad.is_sparse: raise RuntimeError("Adam does not support sparse gradients") 

				state = self.state[p]

				# State initialization 
				if len(state) == 0:
					state["step"] = 0
					# Momentum: Exponential moving average of gradient values 
					state["exp_avg"] = torch.zeros_like(p.data) 
					# RMS prop componenet: Exponential moving average of squared gradient values 
					state["exp_avg_sq"] = torch.zeros_like(p.data) 

				exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"] 
				beta1, beta2 = group["betas"] 
				state["step"] += 1

				if group['weight_decay'] != 0: 
					grad = grad.add(p.data, alpha=group['weight_decay']) 

				# Decay the first and second moment running average coefficient
				exp_avg.lerp_(grad, 1 - beta1) # momentum
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1-beta2) # rms
			
				bias_correction1 = 1 - beta1 ** state["step"] 
				bias_correction2 = 1 - beta2 ** state["step"] 

				step_size = group["lr"] / bias_correction1
				bias_correction2_sqrt = math.sqrt(bias_correction2)

				denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

				p.data.addcdiv_(exp_avg, denom, value=-step_size)  
                
				if group["fixed_sign"]:
					flip_mask = init_weights[i].sign()*p.data.sign()<0
					p.data[flip_mask] = 0
    

class NeuronalData(torch.utils.data.Dataset):
  def __init__(self, inputs, labels):
    self.inputs = inputs
    self.labels = labels

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    input_data = self.inputs[idx]
    label = self.labels[idx]
    return input_data, label