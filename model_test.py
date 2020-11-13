import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTM import LSTM


DATA_DIR = "./data_process/int_1_len_24_91"
MODEL_PATH = './model/tt'
'''
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, output_size=2, num_layers=3):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            # dropout=0.5
        )
        self.linear = torch.nn.Linear(self.hidden_size, output_size)

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
        )

    def forward(self, input):
        # 输入input x:(seq_len, batch, input_size)
        # seq_len:句长，这里不可能为1,论文有问题
        # batch:一次传入数据的量
        # input_size:单词向量长度，即输入量长度
        lstm_out, hidden = self.lstm(input)
        logits = self.linear(lstm_out[-1])
        class_scores = F.log_softmax(logits, dim=1)
        return class_scores

    def get_accuracy(self, logits, target):
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        print(corrects)
        accuracy = 100.0 * corrects / len(target)
        return accuracy.item()
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev_X_preprocessed_data = os.path.join(DATA_DIR, "DEV_X_1.npy")
dev_Y_preprocessed_data = os.path.join(DATA_DIR, "DEV_Y_1.npy")

dev_X = torch.from_numpy(np.load(dev_X_preprocessed_data)).type(torch.Tensor).to(device)
dev_Y = torch.from_numpy(np.load(dev_Y_preprocessed_data)).type(torch.Tensor).to(device)

model = torch.load(MODEL_PATH)
model.eval()

'''
X_local_validation_minibatch, y_local_validation_minibatch = (
    dev_X[i * batch_size: (i + 1) * batch_size, ],
    dev_Y[i * batch_size: (i + 1) * batch_size, ],
)
'''

def test_get_accuracy(logits, target):
    out_index = torch.max(logits.cpu().data, 1)[1].numpy()
    prob = torch.max(logits.cpu().data, 1)[0].numpy()
    # out_index[np.where(prob < 0.85)] = 0
    cor = torch.from_numpy(out_index)
    target = target.cpu()
    corrects = (
        cor.view(target.size()).data == target.data
    ).sum()
    print(corrects)
    accuracy = 100.0 * corrects / len(target)
    return accuracy.item()

#TODO:Xseq?
dev_X = dev_X.permute(1, 0, 2)
dev_Y = torch.max(dev_Y, 1)[1]

y_pred = model(dev_X)
acc = model.get_accuracy(y_pred, dev_Y)
print(acc)
