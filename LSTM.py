import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dataloader


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
        # input_size:单词向量长度，即输入量长度
        lstm_out, hidden = self.lstm(input)
        logits = self.linear(lstm_out[-1])
        class_scores = F.log_softmax(logits, dim=1)
        return class_scores

    def get_accuracy(self, logits, target):
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
