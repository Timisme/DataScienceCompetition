Argparser 

共有四個參數

1. --training | default: training_data.csv
2. --temp_data | default: temp_forecast.json
3. --daily_temp_dir | default: data/daily_data/
4. --output | default: submission.csv

** data 資料夾 應與app.py在同一directory，而daily_data的資料夾在 data 資料夾內。


Method Description 

1. Data Analysis 

* 溫度

	對於備載容量的預測最值觀來講溫度會是主要影響之一，所以我找了2019年以來的各地日均溫，並將各地的溫度做平溫得到台灣該日的溫度均溫(僅考慮大縣市，如離島或者玉山等地不考慮)，並將日均溫與日備載容量分別做MinMax Normalization後得到相關係數0.49，相關性甚高，所以決定以溫度作為預測因子

* 時間因子

	將備載容量畫圖可以發現時間序列呈明顯週期性，剛好上學期有上完統計系的時間序列分析的課，所以就選擇利用SARIMA模型對備載容量時間序列進行配適


2. Model 

* 溫度

	探討溫度與備載容量的關係，對其畫出散布圖後發現兩者呈正向相關，雖然線性相關程度不是非常明顯(有點像指數函數)，為了簡化性而提高預測能力，所以選擇了Slearn中的LinearRegression model進行溫度與備載容量的配適

* ----時間序列----

首先對備載容量做一階差分，使得其時間序列平穩，並做ADFuller test檢定，p值為8*e-18，遠小於0.05，故一階插分後序列呈平穩。

接著對平穩序列分別做ACF和PACF圖，發現ACF前兩步明顯超過信賴界，故ARIMA部分選用MA= 2，且發現ACF圖中每七天有一次週期產生，故對其配飾SARIMA週期部分7步差分；PACF部分也發現前5步皆超過信賴界，故ARIMA部分配飾(5,1,2)，週期部分配飾(0,1,1,7)，配適結果係數皆為顯著。

3. 預測

分別得到溫度對備載容量的回歸模型及備載容量的SARIMA模型後，將兩者的預測值相加除2取平均，則得到最後的預測備載容量。故此預測不但考慮溫度對備載容量的影響，也考慮了資料序列的時間性(與前幾步相關與週期性)，實驗後發現兩者共同考慮後對於預測準確度有顯著的提升(RMSE顯著下降)。





