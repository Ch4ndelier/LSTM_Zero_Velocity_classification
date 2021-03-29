import torch
import torch.nn.functional as F


class LSTM_with_Attention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, output_size=2, num_layers=3):
        super(LSTM_with_Attention, self).__init__()
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

    def attention(self, lstm_output, final_state):
        # lstm_out : (seq_len, batch, num_directions * hidden_size)
        lstm_output = lstm_output.permute(1, 0, 2)
        # lstm_out : (batch, seq_len, num_directions * hidden_size)
        # final state :(num_layers * num_directions, batch_size, hidden_size)
        merged_state = torch.cat([s for s in final_state], 1)
        # merged_state : (1, batch_size, hidden_size * num_layers * num_directions)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # merged_state : (batch_size, hidden_size * num_layers * num_directions, 1)
        weights = torch.bmm(lstm_output, merged_state)
        # TODO:what if num_layer!= 1
        # weights:(batch, seq_len, 1)
        weights = F.softmax(weights.squeeze(2), dim = 1).unsqueeze(2)
        # weights:(batch, seq_len, 1)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)
        # return shape: batch, hidden_size

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
        )

    def forward(self, input):
        # 输入input x:(seq_len, batch, input_size)
        # input_size:单词向量长度，即输入量长度
        # seq_len:一个拥立（句子）的词长度
        lstm_out, (hidden, cell) = self.lstm(input)
        # lstm_out : (seq_len, batch, num_directions * hidden_size)
        # hidden and cell:(num_layers * num_directions, batch_size, hidden_size)

        attn_output = self.attention(lstm_out, hidden)
        logits = self.linear(attn_output)

        class_scores = F.log_softmax(logits, dim=1)
        return class_scores

    def get_accuracy(self, logits, target):
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
