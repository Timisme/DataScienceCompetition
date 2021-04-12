Argparser 

共有四個參數

1. --training | default: training.csv
2. --testing | default: testing.csv
3. --output | default: output.csv

** data 資料夾 應與app.py在同一directory

Method Description 


1. Model 

本次利用LSTM模型作為模型的預測，方法為將training data以seq_len = 10 將每十筆作為一個chunk，input即形成一個size為 (batch_size, seq_len, 4)的tensor，並以15個epoch訓練模型後，再進行預測。

2. 判斷買賣

利用IBM的資料判斷歸納，發現每日報酬率有3成的機率超過1.4%，也有三成的機率每日報酬率低於-1.15%，因為資料為短期資料，故將alpha設為0.3，故可判斷五日內IBM的open price有三成機率大於或小於上述的值，所以若得到下一天的預測後，就會判斷該預測值是否大於前五日1.4%或小於-1.15%，若大於1.4%但手上有持股，則賣出；若小於-1.15%，手上沒有持股，則買進，過程中不做任何空手賣空動作。