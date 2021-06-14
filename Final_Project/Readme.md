* Google Doc Link: https://docs.google.com/document/d/1oeVU5zkLxag6IFDKObn9NGcZT1rrjgMJSmNucOMx950/edit

# 注意事項
1. data folder裡須包含以下檔案
* model.pt
### 前處理後資料：
* train_X1.parquet
* train_X2.parquet
* test_X.parquet 
* test_order_id.npy 
* train_y.npy 
2. 直接執行main.py即可，無須輸入任何參數
3. 前半段會生成訓練和測試資料，約需1分鐘
4. 後段會載入訓練完成之model，並進行測試，過程需20-40分鐘。
5. 預測完成即會產生submission.csv檔
