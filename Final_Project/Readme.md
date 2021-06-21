* Google Slide(主要): https://docs.google.com/presentation/d/1Em6mkc29eKzcUGSw-BfuNTY74Dn8AMEsC2Q3691vYPo/edit#slide=id.p
* Google Doc Link: https://docs.google.com/document/d/1oeVU5zkLxag6IFDKObn9NGcZT1rrjgMJSmNucOMx950/edit

# 注意事項

#### 呈現以google slide為主

### 資料
data folder裡須包含以下檔案

* train_cat.parquet
* train_dense.parquet
* test_cat.parquet 
* test_dense.parquet 
* order_ids.npy 
* train_y.npy 

### 執行方式

#### 直接執行main.py即可，無須輸入任何參數

* 前半段會讀取訓練和測試資料，約需10秒
* 後段會載入訓練完成之model，並進行測試
* 考慮計算資源和時間，僅設置訓練過程為兩個epoch，約需10-20分鐘。
* 若計算資源不足可在main.py降低batch_size[default 256]
* 預測完成即會產生submission.csv檔

### 其他py檔用途

* my_dataset.py 將前處理資料轉成torch中dataset型態
* Preprocess.py 將原始資料進行前處理並生成前處理資料
* model.py 模型架構
* utils.py 包含輔助functions
