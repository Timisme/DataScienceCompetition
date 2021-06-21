import numpy as np 
import pandas as pd 
from Preprocess import get_data
import pyarrow.parquet as pq 

def load_data():
	'''load parquet from data'''

	train_X1= pq.read_table('./data/new_train_X1.parquet').to_pandas()
	train_X2= pq.read_table('./data/new_train_X2.parquet').to_pandas()
	test_X= pq.read_table('./data/test_X.parquet').to_pandas()

	train_X = pd.concat([train_X1, train_X2], axis= 0)

	train_y = train_X['reordered'].fillna(0)
	order_ids = test_X['order_id']
	train_X.drop(columns= ['order_id', 'user_id', 'reordered'], inplace= True)
	test_X.drop(columns= ['order_id', 'user_id'], inplace= True)

	dense_features = [
			'days_since_prior_order_now',
			'add2cart_mode',
			'days_since_mean',
			'days_since_std',
			'order_count',
			'user_reorder_ratio',
			'user_product_ratio',
			'user_order_size',
			'user_dep_ratio',
			'user_aisle_ratio']

	train_dense, test_dense = train_X[dense_features], test_X[dense_features]
	train_cat, test_cat = train_X.drop(columns= dense_features), test_X.drop(columns= dense_features)

	'''標準化連續特徵'''
	scaler = MinMaxScaler()
	train_size = len(train_dense)
	dense = pd.concat([train_dense, test_dense], axis= 0)
	dense = scaler.fit_transform(dense)
	train_dense, test_dense = dense[:train_size, :], dense[train_size:, :] 

	'''cat fields'''
	cat = pd.concat([train_cat, test_cat], axis= 0)
	# cat_fields = [cat['product_id'].max()+1] + [cat[col].nunique() for col in cat.columns[1:]]
	cat_fields = [49689, 7, 24, 7, 24, 2, 135, 22]
	'''num_contns'''
	num_contns = train_dense.shape[1]

	'''convert to array'''
	train_cat, test_cat = train_cat.to_numpy(), test_cat.to_numpy()

	print(cat_fields)
	print(num_contns)
	return train_cat, train_dense, train_y, test_cat, test_dense, cat_fields, num_contns

def split2file():
	# prior = pd.read_parquet('./data/order_products__prior.parquet')

	train_X = pd.read_parquet('./data/new_train_X.parquet')
	size= len(train_X)
	train_X1= train_X.iloc[:int(0.5*size), :]
	train_X2= train_X.iloc[int(0.5*size):, :]
	train_X1.to_parquet('./data/new_train_X1.parquet')
	train_X2.to_parquet('./data/new_train_X2.parquet')


if __name__ == '__main__':
	split2file()