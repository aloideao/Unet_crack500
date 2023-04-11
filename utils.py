import torch
from torch import nn
from dataset import crackdataset
from torch.utils.data import DataLoader
import os
def get_loader(train_path,test_path,batch_size,num_workers,pin_memory):
    train_dataset=crackdataset(train_path)
    test_dataset=crackdataset(test_path,False)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
    return train_loader,test_loader

def save_latest(state,path=os.getcwd(),filename='latest_model.pth'):
  filepath=os.path.join(path,filename)
  torch.save(state,filepath)
def save_best(state,path=os.getcwd(),filename='best.pth'):
  filepath=os.path.join(path,filename)
  torch.save(state,filepath)