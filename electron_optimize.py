#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import scipy
import random
import os
import shutil
import gc
import sys
import uuid
import functools
import gc
import json
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable
from ax import *
from ax.plot.scatter import plot_fitted
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.stats.statstools import agresti_coull_sem
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter


# In[18]:


#Setting reproducability
manualSeed = 158138
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)


torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


FIRST_ARM=5
IT_ARM=3
ITRATION=5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[19]:


#Your dataset class
class MJDataset(Dataset):

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.wfin = np.load("train_input.npy")
        self.wfout = np.load("train_labels.npy")
        self.scaler = StandardScaler()
        self.scaler.fit(np.concatenate([self.wfin, self.wfout],axis=0))
        self.wfin = self.scaler.transform(self.wfin)
        self.wfout = self.scaler.transform(self.wfout)
        self.wfsize = self.wfin.shape[-1]


    def __len__(self):
        return len(self.wfout)
    # @torchsnooper.snoop()
    def __getitem__(self, idx):
        return self.wfin, self.wfout[idx]


# In[20]:


class Model(nn.Module):
    def __init__(self,conv1,conv2,conv3,conv4,conv5,fc1,fc2,fc3,fc4):
        super(Model, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, conv1, kernel_size=5, padding=2), # 1 * 2016 -> conv1 * 2016
            nn.BatchNorm1d(conv1),
            nn.ReLU(),
            nn.MaxPool1d(2)) # conv1 * 2016 -> conv1 * 1008
        self.layer2  = nn.Sequential(
            nn.Conv1d(conv1, conv2, kernel_size=5, padding=2), # conv1 * 1008 -> conv2 * 1008
            nn.BatchNorm1d(conv2),
            nn.ReLU(),
            nn.MaxPool1d(2)) # conv2 * 1008 -> conv2 * 504
        self.layer3  = nn.Sequential(
            nn.Conv1d(conv2, conv3, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv3), # conv2 * 504 -> conv3 * 504
            nn.ReLU(),
            nn.MaxPool1d(2)) # conv3 * 504 -> conv3 * 252
        self.layer4 = nn.Sequential(
            nn.Conv1d(conv3, conv4, kernel_size=5, padding=2), # conv3 * 252 -> conv4 * 252
            nn.BatchNorm1d(conv4),
            nn.ReLU(),
            nn.MaxPool1d(2)) # conv4 * 252 -> conv4 * 126
        self.layer5 = nn.Sequential(
            nn.Conv1d(conv4, conv5, kernel_size=5, padding=2), # conv4 * 126 -> conv5 * 126
            nn.BatchNorm1d(conv5),
            nn.ReLU(),
            nn.MaxPool1d(2)) # conv5 * 126 -> conv5 * 63
        self.fc1 = nn.Linear(63*conv5, fc1)
        self.fc2 = nn.Linear(fc1,fc2)
        self.fc3 = nn.Linear(fc2,fc3)
        self.fc4 = nn.Linear(fc3,fc4)
        self.fc5 = nn.Linear(fc4,1)
    
    #@torchsnooper.snoop()
    # defines how an input tensor flows thru the network
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = torch.sigmoid(out)
        #for i in range(0,out.size()[0]):
            #scored = out[i].item()
            #if (scored > 0.7):
                #plt.plot(range(0,len(x[i].numpy()[0])),x[i].numpy()[0])
        return out


# In[21]:


def train(params):
    #Reconstructing original data
    train_set  = torch.Tensor(np.load("/project/projectdirs/majorana/users/stew314/train_set.npy"))
    train_labels = torch.Tensor(np.load("/project/projectdirs/majorana/users/stew314/train_labels.npy"))
    dataset = torch.utils.data.TensorDataset(train_set,train_labels)

    print(params)

    NUM_EPOCHS = params["num_epoch"]
    batch_size= params["batch_size"]
    LEARNING_RATE =10**(params["learning_rate"])
    ntokens = params["ntokens"]
    emsize = 2*params["nhead"] * params["emmultiplier"]
    nhid = params["nhid"]
    nlayers = params["nlayers"]
    nhead = 2*params["nhead"]
    dropout = params["dropout"]
    conv1 = params["conv1"]
    conv2 = params["conv2"]
    conv3 = params["conv3"]
    conv4 = params["conv4"]
    conv5 = params["conv5"]
    fc1 = params["fc1"]
    fc2 = params["fc2"]
    fc3 = params["fc3"]
    fc4 = params["fc4"]
    optimizer_choice = params["optimizer"]

    validation_split = .3
    shuffle_dataset = True
    random_seed= 42222
    indices = np.arange(len(dataset))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    split = int(validation_split*len(dataset))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,  drop_last=True)

    classifier = Model(conv1,conv2,conv3,conv4,conv5,fc1,fc2,fc3,fc4)
    print("#params", sum(x.numel() for x in classifier.parameters()))
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier.to(DEVICE)

    # criterion = UnsupervisedCELoss()
    criterion = nn.L1Loss()
    criterion = criterion.to(DEVICE)

#     optimizer = torch.optim.RMSprop(
#         classifier.parameters(),
#         momentum=params["momentum"],
#         lr=LEARNING_RATE)
    if(optimizer_choice == "Adam"):
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=LEARNING_RATE)
    elif(optimizer_choice == "SGD"):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
            classifier.parameters()),
            lr=LEARNING_RATE)
    elif(optimizer_choice == "RMSprop"):
        optimizer = torch.optim.RMSprop(classifier.parameters(),
            lr=LEARNING_RATE)
        
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-4,cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,threshold=0.0002,threshold_mode='abs')
    loss_array = []
    for epoch in range(NUM_EPOCHS):
        for i, (wf,wfsmooth) in enumerate(train_loader):
            classifier.train()
            wf = wf.to(DEVICE).float()
            wfsmooth = wfsmooth.to(DEVICE).float()

            outputs  = classifier(wf)
            loss = criterion(outputs.squeeze(-1),wfsmooth)

            loss.backward()
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient
            
            scheduler.step()
            
            if (i % 100 == 0):
                print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                    epoch+1, NUM_EPOCHS, i+1, len(train_loader),
                    loss.item(), end=""))
        # scheduler.step(loss)
        if epoch>NUM_EPOCHS-5:
            loss_array.append(loss.item())
    loss_array = np.array(loss_array)
    print("Average Loss: %.4f"%(np.average(loss_array[loss_array!=np.max(loss_array)])))
    return -np.average(loss_array[loss_array!=np.max(loss_array)])


# In[22]:


#List of Parameters
p1 = FixedParameter(name="num_epoch", value=3, parameter_type=ParameterType.INT)
p2 = RangeParameter(name="batch_size", lower=8, upper=20, parameter_type=ParameterType.INT)
#p3  = RangeParameter(name="learning_rate", lower=-5.0, upper=-3.0, parameter_type=ParameterType.FLOAT)
p3  = FixedParameter(name="learning_rate", value=-4.0, parameter_type=ParameterType.FLOAT)
p4 = RangeParameter(name="ntokens", lower=3, upper=12, parameter_type=ParameterType.INT)
p5 = RangeParameter(name="emmultiplier", lower=3, upper=10, parameter_type=ParameterType.INT)
p6 = RangeParameter(name="nhid", lower=64, upper=196, parameter_type=ParameterType.INT)
p7 = RangeParameter(name="nlayers", lower=2, upper=20, parameter_type=ParameterType.INT)
p8 = RangeParameter(name="nhead", lower=1,upper=15, parameter_type=ParameterType.INT)
p9  = RangeParameter(name="dropout", lower=0.0, upper=0.7, parameter_type=ParameterType.FLOAT)
p10 =  ChoiceParameter(name="optimizer", values=["Adam","SGD","RMSprop"], parameter_type=ParameterType.STRING)
p11 = ChoiceParameter(name="conv1", values=[16,32,64], parameter_type=ParameterType.INT)
p12 = ChoiceParameter(name="conv2", values=[32,64,128], parameter_type=ParameterType.INT)
p13 = ChoiceParameter(name="conv3", values=[64,128,256], parameter_type=ParameterType.INT)
p14 = ChoiceParameter(name="conv4", values=[128,256,512], parameter_type=ParameterType.INT)
p15 = ChoiceParameter(name="conv5", values=[256,512,1024], parameter_type=ParameterType.INT)
p16 = ChoiceParameter(name="fc1", values=[512,256,128], parameter_type=ParameterType.INT)
p17 = ChoiceParameter(name="fc2", values=[256,128,64], parameter_type=ParameterType.INT)
p18 = ChoiceParameter(name="fc3", values=[128,64,32], parameter_type=ParameterType.INT)
p19 = ChoiceParameter(name="fc4", values=[64,32,16], parameter_type=ParameterType.INT)


# In[23]:


class cd:
    '''
    Context manager for changing the current working directory
    '''
    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class BoothMetric(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        auc_result = trial.run_metadata["auc"]
        index = 0
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": auc_result[index],
                "sem": 0.0,
                "trial_index": trial.index
            })
            index += 1
        return Data(df=pd.DataFrame.from_records(records))
        
class MyRunner(Runner):
    def __init__(self):
        '''
        nothing
        '''

    def run(self, trial):
        arm_result = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            auc = train(params)
            arm_result.append(float(auc))
        return {"name": str(trial.index), "auc": arm_result}


# In[ ]:



search_space = SearchSpace(
    parameters=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19],
)#make sure to all all defined parameters here

experiment = Experiment(
    name="hyper_parameter_optimization",
    search_space=search_space,
)
optimization_config = OptimizationConfig(
    objective = Objective(
        metric=BoothMetric(name="booth"), 
        minimize=False,
    )
)

experiment.optimization_config = optimization_config

sobol = Models.SOBOL(search_space=experiment.search_space)
generator_run = sobol.gen(FIRST_ARM)

experiment.runner = MyRunner()
experiment.new_batch_trial(generator_run=generator_run)

experiment.trials[0].run().mark_completed()
data = experiment.fetch_data()

for i in range(1, ITRATION):

    data = experiment.fetch_data()
    gpei = Models.GPEI(experiment=experiment, data=data)
    generator_run = gpei.gen(IT_ARM)
    experiment.new_batch_trial(generator_run=generator_run)
    experiment.trials[i].run().mark_completed()
    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    print(best_arm)
    json_field = best_arm.parameters
    json_field["improvement"] = df['mean'].max()
    with open('data_alpha_v2.json', 'w') as fp:
        json.dump(json_field, fp)
    df.to_json(r'arms_alpha_v2.json')


# In[ ]:




