import numpy as np 
import pandas as pd 
from Preprocess import get_data
import pyarrow.parquet as pq 

def load_data(from_file= True):
	if from_file:
		'''load parquet from data'''
		# train_X1= pd.read_parquet('./data/train_X1.parquet', engine='pyarrow')
		# train_X2= pd.read_parquet('./data/train_X2.parquet', engine='pyarrow')
		# test_X= pd.read_parquet('./data/test_X.parquet', engine='pyarrow')
		
		train_y= np.load('./data/train_y.npy') # allow_pickle=True Series
		train_X1= pq.read_table('./data/train_X1.parquet').to_pandas()
		train_X2= pq.read_table('./data/train_X2.parquet').to_pandas()
		test_X= pq.read_table('./data/test_X.parquet').to_pandas()
		test_order_id= np.load('./data/test_order_id.npy')
		train_X= pd.concat([train_X1, train_X2], axis= 0)
	else:
		train_X, train_y= get_data(train_bool= True)
		test_X, test_order_id= get_data(train_bool= False)
	return train_X, train_y, test_X, test_order_id

def split2file():
	# prior = pd.read_parquet('./data/order_products__prior.parquet')
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


# if __name__ == '__main__':

	# train_X, train_y, test_X, test_order_id= load_data(from_file= True)
	# print(train_X)
	# import numpy as np 
	# import pandas as pd 
	# import torch
	# from sklearn.preprocessing import OneHotEncoder

	# train_X, train_y, test_X, test_order_id= load_data(from_file= True)

	# train_dense_cols= ['train_days_since_prior_order', 'prior_order_count', 'prior_dspo_mean', 'prior_dspo_var', 'user_dep_ratio', 'user_aisle_ratio']
	# test_dense_cols= ['test_days_since_prior_order', 'prior_order_count', 'prior_dspo_mean', 'prior_dspo_var', 'user_dep_ratio', 'user_aisle_ratio']

	# train_X_cat= train_X.drop(train_dense_cols, axis= 1).to_numpy()
	# train_X_dense= train_X[train_dense_cols].to_numpy()
	# # train_y= train_y.to_numpy()

	# test_X_cat= test_X.drop(test_dense_cols, axis= 1).to_numpy()
	# test_X_dense= test_X[test_dense_cols].to_numpy()

	# fields = [len(np.unique(train_X_cat[:, i])) for i in range(train_X_cat.shape[1])] + [len(train_dense_cols)]

	# encoder= OneHotEncoder(sparse= True, handle_unknown='ignore')
	# enc_fitted= encoder.fit(train_X_cat)

	# transformed= enc_fitted.transform(train_X_cat)
	# print(transformed.tocsr()[1, :].todense())