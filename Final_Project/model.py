import torch 
import torch.nn as nn

class DeepFM(nn.Module):
  def __init__(self, cat_fields, num_contns, k, hidden_dims, dropout, n_class, sparse= True):
    super(DeepFM, self).__init__()
    self.cat_fields = cat_fields
    self.num_contns = num_contns 
    self.num_cat = len(cat_fields)
    self.k = k 
    self.hidden_dims = hidden_dims
    self.dropout= nn.Dropout(p=dropout)

    """Linear"""
    # if num_contns != 0:
    self.fm_1st_dense = nn.Linear(num_contns, 1)
    self.fm_1st_cat = nn.ModuleList([nn.Embedding(voc_size, 1, sparse= sparse) for voc_size in cat_fields])

    """embedding"""
    self.embedding_layer = nn.ModuleList([nn.Embedding(voc_size, k, sparse= sparse) for voc_size in cat_fields])
    
    """DNN"""
    layers = []
    input_dim = k * len(cat_fields) + num_contns
    # self.fc_3rd_dense = nn.Linear(num_contns, input_dim) #將contns轉成input_dim過dnn

    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(input_dim, hidden_dim))
      layers.append(nn.BatchNorm1d(hidden_dim))
      layers.append(nn.ReLU(inplace=True))
      layers.append(self.dropout)
      input_dim = hidden_dim
    
    layers.append(nn.Linear(hidden_dims[-1], n_class))
    self.dnn = nn.Sequential(*layers)
    

  def Dense_Embedding(self, X_cat):
    # (batch_size, num_cat)
    cat2dense = [embed(X_cat[:, i].unsqueeze(dim= 1)) for i, embed in enumerate(self.embedding_layer)] # [batch_size, k]
    cat2dense = torch.cat(cat2dense, dim= 1) #[batch_size, num_cat, k]
    return cat2dense

  
  def FM(self, X): # [batch_size, num_cat, k]
    sum_of_square = torch.sum(X, dim= 1)**2 #[n, k]
    square_of_sum = torch.sum(X**2, dim= 1)
    ix = sum_of_square - square_of_sum 
    FM_out = 0.5 * torch.sum(ix, dim= 1, keepdim= True) # [n, 1] 
    return FM_out
  
  def forward(self, X_cat, X_dense):

    '''1st'''
    X_cat_1st = [embed(X_cat[:, i].unsqueeze(dim= 1)) for i, embed in enumerate(self.fm_1st_cat)] # [batch_size, 1]
    X_cat_1st = torch.cat(X_cat_1st, dim= 1) # [batch_size, num_cat, 1]
    X_cat_1st = torch.sum(X_cat_1st, dim= 1, keepdim= False)
    X_dense_1st = self.fm_1st_dense(X_dense)
    y_1st = X_cat_1st + X_dense_1st

    '''2nd'''
    X_cat2dense = self.Dense_Embedding(X_cat) # [batch_size, num_cat, k]
    FM_y = self.FM(X_cat2dense)

    '''3rd'''
    X_cat_flatten = torch.flatten(X_cat2dense, start_dim= 1, end_dim= 2) # [batch_size, num_cat*k]
    # X_dense_flatten = self.fc_3rd_dense(X_dense)
    X_flatten = torch.cat([X_cat_flatten, X_dense], dim= 1)
    DNN_y = self.dnn(X_flatten) # [batch_size, num_cat*k]

    y = y_1st + FM_y + DNN_y

    return y 