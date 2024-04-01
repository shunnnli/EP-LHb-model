# Model comparisons for different models
# Shun Li, 03/08/24

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from EPLHb import EPLHb, gd, adam

import numpy as np
import pickle
from datetime import date

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# Downloading MNIST data
train_data = datasets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)
test_data = datasets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())

# Loading the data
batch_size = 100 # the size of input data took for one iteration
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size,
                                           shuffle = True, generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size,
                                          shuffle = False, generator=torch.Generator(device=device))

train_data.train_data.to(torch.device(device))  # put data into GPU entirely
train_data.train_labels.to(torch.device(device))


# Define conditions
LHb_network = [0, 0.2, 0.4, 0.6, 0.8, 1]
EP_LHb = ['random','dales-law']
LHb_DAN = ['real','mixed','dales-law']
update_methods = ['corelease','fixed-sign']

# Define network basic properties
EP_size = 784 # img_size = (28,28) ---> 28*28=784 in total
LHb_size = 500 # number of nodes at hidden layer
DAN_size = 10 # number of output classes discrete range [0,9]
num_epochs = 10 # number of times which the entire dataset is passed throughout the model
lr = 1e-2 # size of step

prob_EP_to_LHb = 1
prob_LHb_to_DAN = 1

n_networks = 20 # number of networks to train

# Train different networks
training_loss_summary, test_accuracy_summary = {}, {}

for LHb in LHb_network:
    for eplhb in EP_LHb:
        for lhbdan in LHb_DAN:
            for method in update_methods:
                print('LHb: ',LHb, '; EP_LHb:',eplhb,'; LHb_DAN:',lhbdan,'; Method:',method)
                
                # Initialize network-specific loss and accuracy summary
                network_training_loss, network_test_accuracy = [], []

                # Initialize network params
                if LHb == 0: rnn = False
                else: rnn = True
                if method == 'corelease': fixed_sign_update = False
                else: fixed_sign_update = True

                # Train n_networks networks
                for i in range(1,n_networks+1):
                    # Initialize a network
                    net = EPLHb(EP_size,LHb_size,DAN_size,
                                LHb_rnn=rnn,EP_LHb=eplhb,LHb_DAN=lhbdan,
                                prob_EP_to_LHb=prob_EP_to_LHb,prob_LHb_to_LHb=LHb,prob_LHb_to_DAN=prob_LHb_to_DAN)
                    if torch.cuda.is_available(): net.cuda()
                    initial_params = net.record_params(calc_sign=False)
                    training_loss, test_accuracy = [], []

                    # Train on original data
                    optimizer = adam(net.parameters(), lr=lr, fixed_sign=fixed_sign_update)
                    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                    training_loss, test_accuracy = net.train_model(num_epochs,train_loader,optimizer,
                                                    test_loader=test_loader,print_epoch=False,loss='CrossEntropyLoss')

                    network_training_loss.append(training_loss)
                    network_test_accuracy.append(test_accuracy)
                    print('Finished training network %d/%d' %(i,n_networks))

                # Convert list to numpy array
                network_training_loss = np.array(network_training_loss)
                network_test_accuracy = np.array(network_test_accuracy)

                # Store name and stats of network to summary
                network_name = 'RNN'+str(LHb)+'_'+eplhb+'_'+lhbdan+'_'+method
                training_loss_summary[network_name] = network_training_loss
                test_accuracy_summary[network_name] = network_test_accuracy


# Save as pickle file
today = date.today()
filename = '/n/holylabs/LABS/bsabatini_lab/Users/shunnnli/EP-LHb-model/results/MNIST/'+today.strftime("%Y%m%d")+'/model_comparison_'+today.strftime("%Y%m%d")+'.pkl'
print('Saving to',filename)

with open(filename, 'wb') as f:
    data = [training_loss_summary, test_accuracy_summary]
    pickle.dump(data, f)

print('Done')