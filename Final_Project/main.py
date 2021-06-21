import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from my_dataset import custom_dataset
from utils import load_data
from model import DeepFM
import csv
torch.manual_seed(42)

# pyarrow library required

# device= 'cuda' if torch.cuda.is_available() else 'cpu'
device= 'cpu'
print(f'using {device}')
train_cat, train_dense, test_cat, train_y, test_dense, cat_fields, num_contns, order_ids = load_data()

batch_size= 512
lr = 1e-3
n_epoch = 2
k = 8
p = 0.2
hidden_dims = [64, 64]
n_class = 1
threshold = 0.6
step_size = 1
sparse = False
pos_weight = 9

train_dataset = custom_dataset(train_cat, train_dense, train_y, if_y= True)
train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, num_workers=2)

test_dataset = custom_dataset(test_cat, test_dense, if_y= False)
test_loader = DataLoader(test_dataset, batch_size= 512, shuffle= False, num_workers=2)

model= DeepFM(
	cat_fields= cat_fields, 
	num_contns= num_contns, 
	k= k, 
	hidden_dims= hidden_dims, 
	dropout= p, 
	n_class= n_class,
	sparse= sparse).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr= lr)
criterion= nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device= device))
print('model created.')
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma= 0.3, verbose= True)

'''training phase'''
for epoch in range(n_epoch):

	model.train()
	train_loss= 0
	train_score= 0
	val_score= 0
	train_preds, train_gts = [], []
	val_preds, val_gts = [], []

	'''train'''
	for i, (X_cat, X_dense, y) in enumerate(tqdm(train_loader)):
		optimizer.zero_grad()
		output= model(X_cat.to(device), X_dense.to(device))
		loss= criterion(output, y.unsqueeze(dim= 1).to(device))
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		output_boolean = nn.Sigmoid()(output.squeeze(dim= 1))>=threshold
		train_score += sum(output_boolean)
		train_preds += output_boolean.long().tolist()
		train_gts  += y.tolist()

	print(f'\ntrain loss: {round((train_loss/train_size), 6)}| train accu: {round(train_score.item()/train_size, 3)}')

print('training process done!')

preds= []
model.eval()
with torch.no_grad():
	for i, (X_cat, X_dense) in enumerate(tqdm(test_loader)):
		output= model(X_cat.to(device), X_dense.to(device))
		output= nn.Sigmoid()(output) # Careful 
		preds.extend(output.squeeze(dim=1).detach().cpu().numpy()) 

test_X['order_id']= order_ids
test_X['pred']= preds
print('testing process done!')
print('start creating submission file...')
def rule(x):
	if x > 0.6:
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
print('submission file created')