import numpy as np 
import pandas as pd 
# from Preprocess import get_data

def hour2cat(x):
	if (x>=6) & (x<= 12): #早上
		y= 0
	elif (x>12) & (x< 18): #下午
		y= 1
	else:
		y= 2 #晚上
	return y

def split2file():
	prior = pd.read_parquet('./data/order_products__prior.parquet')
	size= len(prior)
	prior1= prior.iloc[:int(0.5*size), :]
	prior2= prior.iloc[int(0.5*size):, :]
	prior1.to_parquet('./data/order_products__prior1.parquet')
	prior2.to_parquet('./data/order_products__prior2.parquet')

	train_X = pd.read_parquet('./data/train_X.parquet')
	size= len(train_X)
	train_X1= train_X.iloc[:int(0.5*size), :]
	train_X2= train_X.iloc[int(0.5*size):, :]
	train_X1.to_parquet('./data/train_X1.parquet')
	train_X2.to_parquet('./data/train_X2.parquet')
	
