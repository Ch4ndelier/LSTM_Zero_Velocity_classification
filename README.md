# LSTM_Zero_Velocity_classification

this is project to classify zero-velocity status.

to be finished

## To get processed data

Link: https://pan.baidu.com/s/1Y34pvkNCPHtdKRNps1niIA   Password: juok

## To start

We suggest you run 

`pip install -r requirements.txt`

Here is an example of config.py:
```
LR = 0.0003
DATA_DIR = "./data_process/int_1_len_24_91_up"
BATCH_SIZE = 600
NUM_EPOCHS = 300
HIDDEN_SIZE = 12
NUM_LAYERS = 2
```

you may write your own config.py to adjust the hyperparameters or use different data
## Scripts

* `process_data_path.py`: Generates the data used in LSTM training

* `train_lstm.py`: Train the LSTM network

* `model_test.py`: Test the trained model