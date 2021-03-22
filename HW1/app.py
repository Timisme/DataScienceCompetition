import os 
import numpy as np 
import pandas as pd
import json
import csv 
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression

def record_generator(file_name):

	drop_loc = ['鞍部', '淡水', '東吉島', '竹子湖', '阿里山', '大武', '成功', '蘭嶼', '梧棲', '日月潭', '金門', '馬祖', '彭佳嶼', '玉山', '新屋']
	with open(f'data/daily_data/{file_name}', 'r') as f: 
		data = [line.strip() for line in f.readlines()]
		record = [np.array(data[i*24].split(','))[[2,3,-1]] for i in range(int(len(data)/24))]
		record_df = pd.DataFrame(record, columns=['loc', 'date', 'temp'])
		record_df.drop((record_df[record_df['temp']=='X'].index), inplace= True)
		record_df.drop(record_df[record_df['loc'].isin(drop_loc)].index, inplace= True)
		
		try:
			record_df['temp'] = record_df['temp'].astype(float)
		except:
			drop_idx = []
			for i, row in record_df.iterrows():
				try:
					float(row['temp'])
				except:
					drop_idx.append(i)
			record_df.drop(drop_idx, inplace= True)
			record_df['temp'] = record_df['temp'].astype(float)
			
	return record_df

def main(training_data= 'training_data.csv', temp_data= 'temp_forecast.json', daily_temp_dir= 'data/daily_data/', output_file= 'submission.csv'):

	with open(f'data/{temp_data}', 'r', encoding='UTF-8') as f:
		forcast = json.load(f)

	forcast_list = forcast['cwbdata']['resources']['resource']['data']['agrWeatherForecasts']['weatherForecasts']['location']

	data_list = []
	for loc_forcast in forcast_list:
		for date_temp in loc_forcast['weatherElements']['MaxT']['daily']:
			data_list.append([date_temp['dataDate'], float(date_temp['temperature'])])
		for date_temp in loc_forcast['weatherElements']['MinT']['daily']:
			data_list.append([date_temp['dataDate'], float(date_temp['temperature'])])

	forcast_df = pd.DataFrame(data_list, columns=['日期', '溫度'])	
	forcast_temp = forcast_df.groupby(['日期'])['溫度'].mean()

	all_files = os.listdir("data/daily_data/")

	all_record = pd.concat([record_generator(file) for file in all_files], axis= 0)
	all_record.drop_duplicates(inplace=True)

	all_record['date'] = all_record['date'].apply(lambda x: x.split(' ')[0])
	all_record['date'] = all_record['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d'))

	y = pd.read_csv(f'data/{training_data}', encoding='big5', parse_dates=['日期'], index_col=['日期'])
	y = y['備載容量']

	'''Temperature regression model'''

	grouped_df = all_record.groupby(['date'])['temp'].mean()
	grouped_normalized = (grouped_df - min(grouped_df))/(max(grouped_df)-min(grouped_df))
	y_normalized = (y['2019':] - min(y['2019':])) / (max(y['2019':])-min(y['2019':]))

	shared_dates = np.intersect1d(grouped_normalized.index, y_normalized.index)
	train_st = datetime(2019, 1, 1)
	train_en = datetime(2021, 3, 15)
	# test_en = datetime(2021, 3, 15)

	train_X = np.array(grouped_normalized[shared_dates][train_st:train_en]).reshape(-1,1)
	train_y = np.array(y_normalized[shared_dates][train_st:train_en]).reshape(-1,1)
	# test_X = np.array(grouped_normalized[shared_dates][train_en+timedelta(days= 1):test_en]).reshape(-1,1)
	# test_y = np.array(y_normalized[shared_dates][train_en+timedelta(days= 1):test_en]).reshape(-1,1)

	reg = LinearRegression().fit(train_X, train_y)

	temp_X = ((forcast_temp - min(grouped_df))/(max(grouped_df)-min(grouped_df))).values.reshape(-1, 1)
	pred_y = (reg.predict(temp_X)*(max(y['2019':])-min(y['2019':])) + min(y['2019':])).reshape(1, -1)[0]
	pred_y = pd.Series(data= pred_y, index= forcast_temp.index)

	print('pred_y:', pred_y)
	'''SARIMA'''

	diff = 1
	first_diff = y['2019':].diff(periods= diff)[diff:]
	y = y.asfreq(freq= pd.infer_freq(y.index))
	train_en = datetime(2021, 3, 21)
	train_data = y['2019':train_en]
	order = (4,1,2)
	seasonal_order = (0, 1, 1, 7)
	SARIMA_model = SARIMAX(train_data, order= order, seasonal_order= seasonal_order)
	SARIMA_model_fit = SARIMA_model.fit()
	forcast_y = SARIMA_model_fit.forecast(8)[1:]

	print('forecast_y:', forcast_y)


	'''Combine'''

	hybrid = 0.5*pred_y.values + 0.5*forcast_y.values

	data = {'dates':['20210323', '20210324', '20210325', '20210326','20210327', '20210328', '20210329'],
	   'operating_reserve(MW)': hybrid}

	with open(f'{output_file}', 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['date', 'operating_reserve(MW)'])

		for i in range(7):
			writer.writerow([data['dates'][i], data['operating_reserve(MW)'][i]])


if __name__ == '__main__':

	import argparse 

	parser = argparse.ArgumentParser()

	parser.add_argument('--training',
		default= 'training_data.csv',
		help= 'input training data file name')

	parser.add_argument('--output',
		default= 'submission.csv',
		help= 'output file name')

	parser.add_argument('--temp_forecast',
		default= 'temp_forecast.json',
		help= 'temperature 7 day forecast file name')

	parser.add_argument('--daily_temp_dir',
		default= 'data/daily_data/',
		help= 'historical temperture file name')

	args = parser.parse_args()

	main(training_data= args.training, temp_data= args.temp_forecast, daily_temp_dir= args.daily_temp_dir, output_file= args.output)
