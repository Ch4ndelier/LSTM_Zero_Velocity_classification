# LSTM_Zero_Velocity_classification

This is project to classify zero-velocity status using the data from accelerometer and gyroscope. 


## To get processed data

Link: https://mega.nz/folder/89QDxKbR  Password: Oy-xsUB8FyFqTjhzy8qYDg

## To start

We suggest you run 

`pip install -r requirements.txt`

`config.py` contains the parameters used in training.You should write your own config.py and put it in the root directory.

Here is an example of config.py:
```
LR = 0.0003
DATA_DIR = "./data_process/int_1_len_24_91_up"
BATCH_SIZE = 600
NUM_EPOCHS = 300
HIDDEN_SIZE = 12
NUM_LAYERS = 2
```

You may write your own config.py to try different hyperparameters or use different data.
## Scripts

* `process_data_path.py`: Generates the data used in LSTM training.

* `train_lstm.py`: Train the LSTM network.

* `model_test.py`: Test the trained model which is saved as "name_you_decide.pkl"

* `data_alignment.py`: To align the data (Our data has lots of problems, we suggest you directly use the data downloaded from Mega, if you need the original data, please contact us by email: ljyuan@bupt.edu.cn).