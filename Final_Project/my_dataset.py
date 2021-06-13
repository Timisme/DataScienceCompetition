import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class custom_dataset(Dataset):
  def __init__(self, X_cat, X_dense, y= None, if_y= False):
    self.X_cat = torch.tensor(X_cat, dtype= torch.float)
    self.X_dense = torch.tensor(X_dense, dtype= torch.float)
    self.if_y= if_y
    if if_y:
      self.y = torch.tensor(y, dtype= torch.float)
  
  def __len__(self):
    return len(self.X_cat)
  
  def __getitem__(self, idx):
    if self.if_y:
      return self.X_cat[idx], self.X_dense[idx], self.y[idx]
    else:
      return self.X_cat[idx], self.X_dense[idx]