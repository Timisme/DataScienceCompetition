import numpy as np 
import pandas as pd 
from Preprocess import get_data
import pyarrow.parquet as pq 
from sklearn.preprocessing import MinMaxScaler

def load_data():
	'''load parquet from data'''
	print('start loading preprocessed data')
	train_cat = pq.read_table('./data/train_cat.parquet').to_pandas().values
	train_dense = pq.read_table('./data/train_dense.parquet').to_pandas().values
	test_cat = pq.read_table('./data/test_cat.parquet').to_pandas().values
	test_dense = pq.read_table('./data/test_dense.parquet').to_pandas().values
	train_y = np.load('./data/train_y.npy')
	order_ids = np.load('./data/order_ids.npy')

	cat_fields = [49689, 7, 24, 7, 24, 2, 135, 22]
	num_contns = 10
	print('data loaded!')
	return train_cat, train_dense, train_y, test_cat, test_dense, cat_fields, num_contns, order_ids

def split2file():
	# prior = pd.read_parquet('./data/order_products__prior.parquet')

	train_X = pd.read_parquet('./data/new_train_X.parquet')
	size= len(train_X)
	train_X1= train_X.iloc[:int(0.5*size), :]
	train_X2= train_X.iloc[int(0.5*size):, :]
	train_X1.to_parquet('./data/new_train_X1.parquet')
	train_X2.to_parquet('./data/new_train_X2.parquet')


# if __name__ == '__main__':
# 	split2file()