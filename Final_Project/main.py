import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from collections import Counter
from my_dataset import custom_dataset
from utils import load_data

# pyarrow library required


# device= 'cuda' if torch.cuda.is_available() else 'cpu'
device= 'cpu'
print(f'using {device}')

train_X, train_y, test_X, test_order_id= load_data(from_file= True)
print('Data Loaded!!!')

train_dense_cols= ['train_days_since_prior_order', 'prior_order_count', 'prior_dspo_mean', 'prior_dspo_var', 'user_dep_ratio', 'user_aisle_ratio']
test_dense_cols= ['test_days_since_prior_order', 'prior_order_count', 'prior_dspo_mean', 'prior_dspo_var', 'user_dep_ratio', 'user_aisle_ratio']

train_X_cat= train_X.drop(train_dense_cols, axis= 1).to_numpy()
train_X_dense= train_X[train_dense_cols].to_numpy()
# train_y= train_y.to_numpy()

test_X_cat= test_X.drop(test_dense_cols, axis= 1).to_numpy()
test_X_dense= test_X[test_dense_cols].to_numpy()

fields = [len(np.unique(train_X_cat[:, i])) for i in range(train_X_cat.shape[1])] + [len(train_dense_cols)]

encoder= OneHotEncoder(sparse= False, handle_unknown='ignore')
enc_fitted= encoder.fit(train_X_cat)

'''test dataloader'''
test_dataset = custom_dataset(test_X_cat, test_X_dense, if_y= False)
test_loader = DataLoader(test_dataset, batch_size= 512, shuffle= False, num_workers=2)

class DeepFM(nn.Module):
  def __init__(self, fields, k= 5, hidden_dims= [16, 16], dropout= 0.2, n_class= 1):
    super(DeepFM, self).__init__()
    self.fields = fields 
    self.k = k 
    self.hidden_dims = hidden_dims
    self.dropout= nn.Dropout(p=dropout)

    """Linear"""
    d = sum(fields)
    self.linear = nn.Linear(d, n_class, bias= False)

    """FM"""
    # self.FM_w = nn.Linear(1, n_class)
    self.embedding_ws = nn.ModuleList()
    for i in fields:
      self.embedding_ws.append(nn.Linear(i, k, bias= False))
    
    """DNN"""
    layers = []
    input_dim = k * len(fields)

    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(input_dim, hidden_dim))
      layers.append(nn.BatchNorm1d(hidden_dim))
      layers.append(nn.ReLU())
      layers.append(self.dropout)
      input_dim = hidden_dim
    
    layers.append(nn.Linear(hidden_dims[-1], n_class))
    self.dnn = nn.Sequential(*layers)

  def Dense_Embedding(self, X):
    es = []
    start= 0
    for i, field in enumerate(self.fields):
      ei = self.embedding_ws[i](X[:, start:start+field]).unsqueeze(dim= 1) # ei: [n, 1, k]
      # ei = torch.matmul(X[:, start:start+field], self.embedding_ws[i]).unsqueeze(dim= 1) # ei: [n, 1, k]
      start += field
      es.append(ei)

    return torch.cat(es, dim= 1) # [n, n_fields, k]  

  
  def FM(self, X):

    sum_of_square = torch.sum(X, dim= 1)**2 #[n, k]
    square_of_sum = torch.sum(X**2, dim= 1)
    ix = sum_of_square - square_of_sum 
    FM_out = 0.5 * torch.sum(ix, dim= 1, keepdim= True) # [n, 1] 
    FM_out = self.dropout(FM_out)
    # return self.FM_w(FM_out)
    return FM_out

  def DNN(self, X):

    X = X.view(-1, self.k * len(self.fields)) # [n, k*n_fields]
    X = self.dnn(X)
    return X
  
  def forward(self, X):

    dense_X = self.Dense_Embedding(X)
    FM_y = self.FM(dense_X)
    DNN_y = self.DNN(dense_X)
    y = self.dropout(self.linear(X)) + FM_y + DNN_y

    # return nn.Sigmoid()(y) # BCELoss
    return y # nn.BCEWithLogitsLoss(pos_weight=9)

"""load model from pt file"""
# print('Loading model...')
# model= torch.load('./data/model.pt', map_location=torch.device(device))

"""Training"""
batch_size= 512
lr = 1e-3
n_epoch = 2
k = 10
p = 0.5
hidden_dims = [64, 64]
n_class = 1

""""""
train_dataset = custom_dataset(train_X_cat, train_X_dense, train_y, if_y= True)
train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, num_workers=2)

test_dataset = custom_dataset(test_X_cat, test_X_dense, if_y= False)
test_loader = DataLoader(test_dataset, batch_size= 512, shuffle= False, num_workers=2)

model= DeepFM(fields= fields, k= k, hidden_dims= hidden_dims, dropout= p, n_class= n_class).to(device)
# model= torch.load('/content/drive/MyDrive/python_data/kaggle/instacart/instacart.pt', map_location=torch.device(device))
optimizer = torch.optim.Adam(model.parameters(), lr= lr)
criterion= nn.BCEWithLogitsLoss(pos_weight=torch.tensor(9, device= device))

for epoch in range(n_epoch):
	model.train()
	total_loss= list()
	for i, (X_cat, X_dense, y) in enumerate(tqdm(train_loader)):
	# for i, (X_cat, X_dense, y) in enumerate(train_loader):
		X_cat_onehot = torch.tensor(enc_fitted.transform(X_cat), dtype= torch.float)
		X= torch.cat([X_cat_onehot, X_dense], dim= 1).to(device)
		optimizer.zero_grad()
		output= model(X)
		loss= criterion(output, y.unsqueeze(dim= 1).to(device))
		loss.backward()
		optimizer.step()
		total_loss.append(loss.item())
	# if i == 200:
	#   break
	print(f'avg loss: {round(np.mean(total_loss), 4)}')

"""Testing Phase"""
print('start test phase...')
preds= []
model.eval()
with torch.no_grad():
	for i, (X_cat, X_dense) in enumerate(tqdm(test_loader)):
		X_cat_onehot = torch.tensor(enc_fitted.transform(X_cat), dtype= torch.float)
		X= torch.cat([X_cat_onehot, X_dense], dim= 1).to(device)
		output= model(X)
		output= nn.Sigmoid()(output) # Careful 
		preds.extend(output.squeeze(dim=1).detach().cpu().numpy())  

test_X['order_id']= test_order_id
test_X['pred']= preds

def rule(x):
	if x >= 0.6:
		return 1 
	else:
		return 0

test_X['pred_binary'] = test_X['pred'].apply(rule)

submission_dict= {}
for i, row in test_X[test_X['pred_binary']==1].iterrows():
	order_id = int(row['order_id'])
	product_id= int(row['product_id'])
	if order_id in submission_dict.keys():
		submission_dict[order_id].append(product_id)
	else:
		submission_dict[order_id] = [product_id]

for order_id in test_X[test_X['pred_binary']==0]['order_id'].unique():
	if order_id in submission_dict.keys():
		pass
	else:
		submission_dict[order_id]= 'None'

with open('submission.csv', 'w', newline='') as csvfile:
	# 建立 CSV 檔寫入器
	writer = csv.writer(csvfile, delimiter=',')

	# 寫入一列資料
	writer.writerow(['order_id', 'products'])

	# 寫入另外幾列資料
	for key, value in submission_dict.items():
		if value == 'None':
			writer.writerow([key, 'None'])
		else:
			value= [str(id) for id in value]
		writer.writerow([str(key), ' '.join(value)])