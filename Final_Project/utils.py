def hour2cat(x):
	if (x>=6) & (x<= 12): #早上
		y= 0
	elif (x>12) & (x< 18): #下午
		y= 1
	else:
		y= 2 #晚上
	return y

if __name__ == '__main__':
	
	import numpy as np 
	import pandas as pd 

	df = pd.read_csv('./data/order_products__prior.csv')
	df.to_pickle('./data/order_products__prior.pkl')
