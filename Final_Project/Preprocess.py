import numpy as np 
import pandas as pd 
from collections import Counter
from utils import hour2cat


def get_data(train_bool= True):

	print('Loading csv files...')
	folder_path = './data/'
	files = ['aisles.csv', 'departments.csv', 'order_products__prior.csv', 'order_products__train.csv', 'orders.csv', 'products.csv']
	prior = pd.read_csv(folder_path+files[2])
	train = pd.read_csv(folder_path+files[3])
	orders = pd.read_csv(folder_path+files[4])
	products = pd.read_csv(folder_path+files[5])
	test = orders[orders['eval_set'] == 'test']
	orders['order_hour_of_day'] = orders['order_hour_of_day'].apply(lambda x: hour2cat(x))
  	
	prior_order = orders[orders['eval_set']=='prior'].drop(columns= ['eval_set'], axis= 1)

	if train_bool:
		print('Generating Training data...')
		train_order = orders[orders['eval_set']=='train'].drop(columns= ['eval_set'], axis= 1)
		train_order.columns = ['train_'+col for col in train_order.columns]
		prior_order.columns = ['prior_'+col for col in prior_order.columns]
		train_prior = pd.merge(train_order, prior_order, left_on= 'train_user_id', right_on='prior_user_id')
		train_X = pd.merge(train_prior, prior, left_on= 'prior_order_id', right_on= 'order_id')
		train_X = pd.merge(train_X, products, left_on= 'product_id', right_on= 'product_id')

		'''針對歷史order資訊做groupby'''
		train_cols = ['train_order_id', 'train_user_id', 'train_order_dow', 'train_order_hour_of_day', 'train_days_since_prior_order', 'product_id', 'aisle_id', 'department_id']
		grouped_train_X = train_X.groupby(train_cols, dropna=False).agg(
			prior_order_count= ('prior_order_id', 'count'), 
	  		prior_dow_mode= ('prior_order_dow', lambda x: Counter(x).most_common(1)[0][0]), 
	  		prior_hod_mode= ('prior_order_hour_of_day', lambda x: Counter(x).most_common(1)[0][0]), 
  			prior_dspo_mean= ('prior_days_since_prior_order', 'mean'),
	  		prior_dspo_var= ('prior_days_since_prior_order', 'var')
	  		).reset_index().fillna(0)

		'''user feature with dep & aisle'''
		train_user_dep_ratio= train_X.groupby(['train_user_id', 'department_id'])['department_id'].count()/train_X.groupby(['train_user_id'])['department_id'].count()
		train_user_aisle_ratio= train_X.groupby(['train_user_id', 'aisle_id'])['aisle_id'].count()/train_X.groupby(['train_user_id'])['aisle_id'].count()
		grouped_train_X['user_dep_ratio'] = train_user_dep_ratio[pd.MultiIndex.from_frame(grouped_train_X[['train_user_id', 'department_id']])].values
		grouped_train_X['user_aisle_ratio'] = train_user_aisle_ratio[pd.MultiIndex.from_frame(grouped_train_X[['train_user_id', 'aisle_id']])].values

		'''Train x合併y'''
		full_X = pd.merge(grouped_train_X, train.rename(columns={'reordered':'label'}), how= 'left', left_on=['train_order_id','product_id'], right_on= ['order_id', 'product_id'])

		'''drop不需要的columns'''
		drop_cols= ['train_user_id', 'train_order_id', 'order_id', 'add_to_cart_order']
		train_X = full_X.drop(columns=drop_cols, axis= 1).fillna(0).iloc[:, :-1]
		train_y = full_X.drop(columns=drop_cols, axis= 1).fillna(0).iloc[:, -1]
		print('training data ok')
		return train_X, train_y
  
	else:
		print('Generating test data...')
		test_order = orders[orders['eval_set']=='test'].drop(columns= ['eval_set'], axis= 1)
		prior_order.columns = ['prior_'+col for col in prior_order.columns]
		test_order.columns = ['test_'+col for col in test_order.columns]
		test_prior = pd.merge(test_order, prior_order, left_on= 'test_user_id', right_on='prior_user_id')
		test_X = pd.merge(test_prior, prior, left_on= 'prior_order_id', right_on= 'order_id')
		test_X = pd.merge(test_X, products, left_on= 'product_id', right_on= 'product_id')

		'''針對歷史order資訊做groupby'''

		test_cols= ['test_order_id', 'test_user_id', 'test_order_dow', 'test_order_hour_of_day', 'test_days_since_prior_order', 'product_id', 'aisle_id', 'department_id']
		grouped_test_X = test_X.groupby(test_cols, dropna=False).agg(
			prior_order_count= ('prior_order_id', 'count'), 
			prior_dow_mode= ('prior_order_dow', lambda x: Counter(x).most_common(1)[0][0]), 
			prior_hod_mode= ('prior_order_hour_of_day', lambda x: Counter(x).most_common(1)[0][0]), 
			prior_dspo_mean= ('prior_days_since_prior_order', 'mean'),
			prior_dspo_var= ('prior_days_since_prior_order', 'var')
			).reset_index().fillna(0)

		test_order_id = test_X['test_order_id'].values

		'''user feature with dep & aisle'''
		test_user_dep_ratio= test_X.groupby(['test_user_id', 'department_id'])['department_id'].count()/test_X.groupby(['test_user_id'])['department_id'].count()
		test_user_aisle_ratio= test_X.groupby(['test_user_id', 'aisle_id'])['aisle_id'].count()/test_X.groupby(['test_user_id'])['aisle_id'].count()

		grouped_test_X['user_dep_ratio'] = test_user_dep_ratio[pd.MultiIndex.from_frame(grouped_test_X[['test_user_id', 'department_id']])].values
		grouped_test_X['user_aisle_ratio'] = test_user_aisle_ratio[pd.MultiIndex.from_frame(grouped_test_X[['test_user_id', 'aisle_id']])].values
		test_X = grouped_test_X.drop(columns=['test_user_id', 'test_order_id'], axis= 1).fillna(0)
		print('test data ok')
		return test_X, test_order_id