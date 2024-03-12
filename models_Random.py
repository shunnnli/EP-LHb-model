# Model comparisons for different models
# Shun Li, 03/08/24

import torch
import torch.nn as nn

from EPLHb import EPLHb, gd, adam, NeuronalData

import numpy as np
from scipy import stats
import pickle
from datetime import date

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# Define conditions
LHb_network = ['MLP','RNN']
initialization = ['random','dales_law']
network_struct = ['real','mixed']
update_methods = ['corelease','fixed_sign']

# Define network basic properties
EP_size = 784 # img_size = (28,28) ---> 28*28=784 in total
LHb_size = 500 # number of nodes at hidden layer
DAN_size = 10 # number of output classes discrete range [0,9]
num_epochs = 200 # number of times which the entire dataset is passed throughout the model
lr = 1e-2 # size of step

prob_EP_to_LHb = 1
prob_LHb_to_LHb = 1
prob_LHb_to_DAN = 1

n_networks = 20 # number of networks to train

label_type = 'digital' # or 'digital'
prob_input_active = 0.05 # probability that an input is active in each context
prob_output_active = 0.125
n_contexts = 5000
prob_EP_flip = 0.05
# generator = torch.Generator(device=device)

# Generate initial random data
print('Generating data...')
rands = torch.rand(n_contexts, EP_size, device=device)
train_data = 1.0*(rands<prob_input_active) - 1.0*(rands>(1-prob_input_active))
rands = torch.rand(n_contexts, device=device)
if label_type == 'analog': train_labels = 2*rands-1
else: train_labels = 1.0*(rands<prob_output_active) - 1.0*(rands>(1-prob_output_active))
train_labels = torch.transpose(train_labels.repeat(DAN_size, 1).squeeze(), 0, 1)

# Randomly select inputs, and flip corresponding labels
print('Flipping data...')
input_mask = torch.rand(EP_size,device=device) < prob_EP_flip
flip_EP = torch.linspace(1,EP_size,EP_size)[input_mask].to(torch.int32)
flip_idx = train_data.nonzero()[torch.isin(train_data.nonzero()[:,1], flip_EP),0].unique()
train_labels_flipped = train_labels.clone()
train_labels_flipped[flip_idx] *= -1
n_flip = flip_idx.shape[0]
print('Flipped percentage: %.3f%%, %d/%d' % (100*n_flip/n_contexts, n_flip, n_contexts))
print('Flipped EP neurons: ' + str(flip_EP.cpu().numpy()))

# Packaged into dataset
batch_size = 100
train_dataset = NeuronalData(train_data,train_labels,device=device)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
flip_dataset = NeuronalData(train_data,train_labels_flipped,device=device)
flip_loader = torch.utils.data.DataLoader(dataset=flip_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

# Train different networks
print('Training networks...')
training_loss_summary, relearn_loss_summary = {}, {}

for LHb in LHb_network:
    for init in initialization:
        for struct in network_struct:
            for method in update_methods:
                print('LHb: ',LHb, '; Initialization:',init,'; Network:',struct,'; Method:',method)
                
                # Initialize network-specific loss and accuracy summary
                network_training_loss, network_relearn_loss = [], []

                # Initialize network params
                if LHb == 'MLP': rnn = False
                else: rnn = True
                if init == 'random': fixed_sign_init = False
                else: fixed_sign_init = True
                if struct == 'real': real_circuit = True
                else: real_circuit = False
                if method == 'corelease': fixed_sign_update = False
                else: fixed_sign_update = True

                # Train n_networks networks
                for i in range(1,n_networks+1):
                    # Initialize a network
                    net = EPLHb(EP_size,LHb_size,DAN_size,
                                LHb_rnn=rnn,fixed_sign=fixed_sign_init,real_circuit=real_circuit,
                                prob_EP_to_LHb=prob_EP_to_LHb,prob_LHb_to_LHb=prob_LHb_to_LHb,prob_LHb_to_DAN=prob_LHb_to_DAN)
                    if torch.cuda.is_available(): net.cuda()

                    training_loss, relearn_loss = [], []

                    # Train on original data
                    optimizer = adam(net.parameters(), lr=lr, fixed_sign=fixed_sign_update)
                    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                    training_loss, _ = net.train_model(num_epochs,train_loader,optimizer,
                                                    print_epoch=False,loss='MSE')

                    # Train on flipped data
                    optimizer = adam(net.parameters(), lr=lr, fixed_sign=fixed_sign_update)
                    relearn_loss, _ = net.train_model(num_epochs,flip_loader,optimizer,
                                                    print_epoch=False,loss='MSE')

                    network_training_loss.append(training_loss)
                    network_relearn_loss.append(relearn_loss)
                    print('Finished training network %d/%d' %(i,n_networks.cpu()))

                # Convert list to numpy array
                network_training_loss = np.array(network_training_loss)
                network_relearn_loss = np.array(network_relearn_loss)

                # Store name and stats of network to summary
                network_name = LHb+'_'+init+'_'+struct+'_'+method
                training_loss_summary[network_name] = network_training_loss
                relearn_loss_summary[network_name] = network_relearn_loss


# Save as pickle file
today = date.today()
filename = '~/code/EP-LHb-model/results/Random/model_comparison_'+today.strftime("%Y%m%d")+'.pkl'
print('Saving to',filename)

with open(filename, 'wb') as f:
    data = [training_loss_summary, relearn_loss_summary]
    pickle.dump(data, f)

print('Done')