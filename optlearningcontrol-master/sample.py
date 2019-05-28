
# coding: utf-8

# Loading the libraries 

# In[18]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np
import matplotlib.pyplot as plt


# Generate Data

# In[19]:


def generate_data_q4(data_size=5000,lim=10):
    
    points = torch.FloatTensor(data_size,2).uniform_(-lim, lim)
    function = torch.FloatTensor(data_size)
    for i in range(data_size):
        function[i]  = points[i,0]**2 + points[i,1]**2 + points[i,0]*points[i,1]    
    return points, function


# In[20]:


points, function = generate_data_q4()


# Data Loader

# In[21]:


batch_size_val = 100
random_seed = 0
test_size = 0.1
num_train = len(points)
indices = list(range(num_train))
split = int(np.floor(test_size * num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

dataset = torch.utils.data.TensorDataset(points, function)

train_loader = torch.utils.data.DataLoader(dataset,
    sampler=train_sampler, batch_size=batch_size_val,
    drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset,
    sampler=valid_sampler,batch_size=batch_size_val, 
    drop_last=True)


# Neural Network Architecture

# In[22]:


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = Net()
print(network)


# In[23]:


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr = 0.00005)


# Train and Test Functions

# In[30]:


def train():

    for i, (inputs, labels) in enumerate(train_loader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print('Training MSELoss: {:.6f},'.format(loss.data))
    train_loss = loss.item()
    
    return train_loss

#---------- Testing Function ----------#

def test():

    for i, (inputs, labels) in enumerate(test_loader, 0):
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print('Testing MSELoss: {:.6f}'.format(loss.data))
    test_loss = loss.data.item()
    
    return test_loss


# Running the Whole Thing

# In[31]:


epch_val = 200
train_loss = np.zeros((epch_val,1))
test_loss  = np.zeros((epch_val,1))

for epoch in range(epch_val):  # loop over the dataset multiple times

    print('------------ Epoch {} ------------'.format(epoch))
    
    train_loss[epoch] = train()
    test_loss[epoch]  = test()
    
print('Finished Training')



epc_c = range(1,epch_val+1)

plt.figure(1)
plt.plot(epc_c,train_loss,'r')
plt.plot(epc_c,test_loss,'b')
plt.xlabel('epochs')
plt.ylabel('MSE Loss')
plt.legend(['Training','Testing'])
#plt.grid(True)
plt.show()

