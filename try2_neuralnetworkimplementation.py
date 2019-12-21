#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env bash
#SBATCH --job-name=torch1
#SBATCH --output=newtorch1.out
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:1
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import math
import numpy as np
import time
import os
import pdb
cuda = True if torch.cuda.is_available() else False
    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)
    
if cuda:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
    
torch.set_num_threads(8)


# In[2]:


class Model(nn.Module):
    def __init__(self, kernel_size, output_sizes, strides):
        super(Model, self).__init__()
        
        self.n_layers = len(output_sizes)-1
        
        module = []
        hiddens = [100, 50]
        padding = 2
        new_h = 200
        new_w = 200
        convStride = 1 
        drop = 0.2
        
#         output_sizes = torch.Tensor(output_sizes)
#         hiddens = torch.Tensor(hiddens)
#         strides = torch.Tensor(strides)
        
        for l in range(self.n_layers):
            module.append(nn.Conv2d(output_sizes[l], output_sizes[l+1], kernel_size=kernel_size, padding=padding))
            module.append(nn.ReLU())
            module.append(nn.MaxPool2d(kernel_size=kernel_size, stride=strides[l]))
            new_h = np.floor((np.floor((new_h-kernel_size+2*padding)/convStride+1))/strides[l])
            new_w = np.floor((np.floor((new_w-kernel_size+2*padding)/convStride+1))/strides[l])
        
        nFeatures = new_h*new_w*output_sizes[-1]*3
            
        self.model_xy = nn.Sequential(*module)
        self.model_yz = nn.Sequential(*module)
        self.model_xz = nn.Sequential(*module)
        
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(int(nFeatures), hiddens[0]),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hiddens[0], hiddens[1]),
            nn.ReLU(),
            nn.Linear(hiddens[1], 1)
        )
        
    def forward(self, x):
        out1 = torch.flatten(self.model_xy(x[:, 0, :, :, :]))
        out1 = out1.view(out1.shape[0], 1)
        out2 = torch.flatten(self.model_xz(x[:, 1, :, :, :]))
        out2 = out2.view(out2.shape[0], 1)
        out3 = torch.flatten(self.model_yz(x[:, 2, :, :, :]))
        out3 = out3.view(out3.shape[0], 1)
        		
        z = torch.cat((out1, out2, out3))
        z = self.classifier(z)
        return z


# In[3]:


output_sizes = [16, 32, 32, 64, 64]
strides = [2, 2, 2, 2]
kernel_size = 7
h = 200
w = 200

model = Model(kernel_size, output_sizes, strides)

criterion = nn.MSELoss()
learning_rate = 0.05

batch_size = 64
n_iters = 6000

# num_epochs = n_iters / (len(X) / batch_size)
# num_epochs = int(num_epochs)


# In[4]:


class Fetcher:
    def __init__(self):
        path = '/ihome/workshops/shm150/DATA'
        os.chdir(path)
        filenames_og = [f for f in os.listdir('./OG/XY/') if f[-3:]=='csv']
        filenames_tsg = [f for f in os.listdir('./TSG/XY/') if f[-3:]=='csv']
        og_filenames = [[os.path.join(path+'/OG/XY/',f),         os.path.join(path+'/OG/XZ/',f),         os.path.join(path+'/OG/YZ/',f)] for f in filenames_og]
        tsg_filenames = [[os.path.join(path+'/TSG/XY/',f),         os.path.join(path+'/TSG/XZ/',f),         os.path.join(path+'/TSG/YZ/',f)] for f in filenames_tsg]
        filenames = og_filenames + tsg_filenames
        
        train_size = int(len(filenames)*0.75)
        self.train = filenames[:train_size]
        self.test = filenames[train_size:]
        self.features = ['charged(side chain can make salt bridges)',                        'Polar(usually participate in hydrogen bonds as proton donnars & acceptors)',                        'Hydrophobic(normally burried inside the protein core)', 'Hydrophobic',                        'Moderate', 'Hydrophillic', 'polar', 'Aromatic', 'Aliphatic', 'Acid',                        'Basic', 'negative charge', 'Neutral', 'positive charge', 'Pka_NH2',                        'P_ka_COOH']
        
    def create_image(self, path_list):
        images = np.ndarray([batch_size, 3, 16, 200, 200])
        labels = np.ndarray([batch_size])
        for i,f_list in enumerate(path_list):
            for per_ind,f in enumerate(f_list):
                df = pd.read_csv(f)
                a_coord = f.split('/')[-2][0].lower()+'_coord'
                b_coord = f.split('/')[-2][1].lower()+'_coord'
                coord_df = df[[a_coord, b_coord]].round().astype('int64')
                img = np.zeros([201, 201])
                for fe_ind,feature in enumerate(self.features):
                    img[coord_df[a_coord].values, coord_df[b_coord].values] = df[feature]
                    images[i, per_ind, fe_ind] = img[:200, :200]
            label = path_list[0][-3]
            if label=='OG':
                labels[i] = 0
            elif label=='TSG':
                labels[i] = 1
                    
        return images, labels

    def load_batch(self):
        path_list = self.train[:batch_size]
        self.train = self.train[batch_size:]
        return self.create_image(path_list)


# In[5]:


def train():
    model = Model(kernel_size, output_sizes, strides).double()
    model.to(device)
    
    model.zero_grad()
    fetcher = Fetcher()
    total_files = len(fetcher.train)
    
    EPOCHS =  int(n_iters/(total_files/ batch_size))
    criterion = nn.MSELoss()
    learning_rate = 0.05
    optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
    avg_loss = 0
    counter = 0
        
    model.train()
    print("Starting the training : ")
    
    for epoch in range(1, EPOCHS+1):
        X, Y = fetcher.load_batch()
        
        for k in range(total_files):
            counter += 1
            X = Tensor(X)
            Y = Tensor(Y)
            out = model(X.to(device).double())
            loss = criterion(out, Y.to(device).double())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, total_files, avg_loss/counter))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/total_files))
    return model  


# In[ ]:


train()


# In[ ]:





# In[15]:


from torchsummary import summary
summary(model, input_size=(16, 200, 200))


# In[2]:


import os
os.listdir('../')


# In[ ]:




