# Model comparisons for different models
# Shun Li, 03/08/24

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from EPLHb import EPLHb, gd, adam

import numpy as np
from scipy import stats
import pickle
from datetime import date

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Downloading MNIST data
train_data = datasets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = datasets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())

# Loading the data
batch_size = 100 # the size of input data took for one iteration
train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size = batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data,batch_size = batch_size,shuffle = False)


# Define conditions
LHb_network = ['MLP','RNN']
initialization = ['random','dales_law']
network_struct = ['real','mixed']
update_methods = ['corelease','fixed_sign']

# Define network basic properties
EP_size = 784 # img_size = (28,28) ---> 28*28=784 in total
LHb_size = 500 # number of nodes at hidden layer
DAN_size = 10 # number of output classes discrete range [0,9]
num_epochs = 20 # number of times which the entire dataset is passed throughout the model
lr = 1e-3 # size of step

prob_EP_to_LHb = 1
prob_LHb_to_LHb = 1
prob_LHb_to_DAN = 1

n_networks = 20 # number of networks to train


# Train different networks
training_loss_summary, test_accuracy_summary = {}, {}

for LHb in LHb_network:
    for init in initialization:
        for struct in network_struct:
            for method in update_methods:
                print('LHb: ',LHb, '; Initialization:',init,'; Network:',struct,'; Method:',method)
                
                # Initialize network-specific loss and accuracy summary
                network_training_loss, network_test_accuracy = [], []

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
                                rnn=rnn,fixed_sign=fixed_sign_init,real_circuit=real_circuit,
                                prob_EP_to_LHb=prob_EP_to_LHb,prob_LHb_to_LHb=prob_LHb_to_LHb,prob_LHb_to_DAN=prob_LHb_to_DAN)
                    initial_params = net.record_params(calc_sign=False)
                    training_loss, test_accuracy = [], []
                    if torch.cuda.is_available(): net.cuda()

                    # Train on original data
                    optimizer = adam(net.parameters(), lr=lr, fixed_sign=fixed_sign_update)
                    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                    training_loss, test_accuracy = net.train_model(num_epochs,train_loader,optimizer,
                                                    test_loader=test_loader,print_epoch=False,loss='CrossEntropyLoss')
                    training_loss.extend(training_loss)
                    test_accuracy.extend(test_accuracy)

                    # Train on flipped data
                    # optimizer = adam(net.parameters(), lr=lr, fixed_sign=fixed_sign_update)
                    # training_loss = net.train_model(num_epochs,flip_loader,optimizer,print_epoch=False)
                    # net_training_loss.extend(training_loss)

                    network_training_loss.append(training_loss)
                    network_test_accuracy.append(test_accuracy)
                    print('Finished training network %d/%d' %(i,n_networks))

                # Convert list to numpy array
                network_training_loss = np.array(network_training_loss)
                network_test_accuracy = np.array(network_test_accuracy)

                # Store name and stats of network to summary
                network_name = LHb+'_'+init+'_'+struct+'_'+method
                training_loss_summary[network_name] = network_training_loss
                test_accuracy_summary[network_name] = network_test_accuracy


# Save as pickle file
today = date.today()
filename = 'Results/model_comparison_'+today.strftime("%Y%m%d")+'.pkl'
print('Saving to',filename)

with open(filename, 'wb') as f:
    data = [training_loss_summary, test_accuracy_summary]
    pickle.dump(data, f)

print('Done')