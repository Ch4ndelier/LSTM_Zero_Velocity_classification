import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

train_X_preprocessed_data = "./data/data_train_input.npy"
train_Y_preprocessed_data = "./data/data_train_target.npy"
dev_X_preprocessed_data = "./data/data_validation_input.npy"
dev_Y_preprocessed_data = "./data/data_validation_target.npy"
test_X_preprocessed_data = "./data/data_test_input.npy"
test_Y_preprocessed_data = "./data/data_test_target.npy"

if (
    os.path.isfile(train_X_preprocessed_data)
    and os.path.isfile(train_Y_preprocessed_data)
    and os.path.isfile(dev_X_preprocessed_data)
    and os.path.isfile(dev_Y_preprocessed_data)
    and os.path.isfile(test_X_preprocessed_data)
    and os.path.isfile(test_Y_preprocessed_data)
):
    print("Preprocessed files exist, deserializing npy files")
    train_X = torch.from_numpy(np.load(train_X_preprocessed_data)).type(torch.Tensor)
    train_Y = torch.from_numpy(np.load(train_Y_preprocessed_data)).type(torch.Tensor)
    print(train_X.type())
    print(np.load(train_X_preprocessed_data).shape)
    print(np.load(train_Y_preprocessed_data).shape)
    exit()
    dev_X = torch.from_numpy(np.load(dev_X_preprocessed_data)).type(torch.Tensor)
    dev_Y = torch.from_numpy(np.load(dev_Y_preprocessed_data)).type(torch.Tensor)

    test_X = torch.from_numpy(np.load(test_X_preprocessed_data)).type(torch.Tensor)
    test_Y = torch.from_numpy(np.load(test_Y_preprocessed_data)).type(torch.Tensor)
else:
    print("raw audio files")

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, output_size = 8, num_layers = 4):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size 
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers
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
        class_scores = F.log_softmax(logits, dim = 1)
        return class_scores

    def get_accuracy(self, logits, target):
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()


batch_size = 35
num_epochs = 20

# Define model
print("Build LSTM model ..")
model = LSTM(
    input_size = 33, #TODO : 6
    hidden_size = 90,
    batch_size = batch_size,
    output_size = 8, #TODO : 2
    num_layers = 4
)
loss_function = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.001)

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("\n Training on GPU")
else:
    print("\n No GPU, training on CPU")

num_batches = int(train_X.shape[0] / batch_size)
num_dev_batches = int(dev_X.shape[0] / batch_size)

val_loss_list, val_accuracy_list, epoch_list = [], [], []

print("Training ...")

for epoch in range(num_epochs):

    train_running_loss, train_acc = 0.0, 0.0

    model.hidden = model.init_hidden()
    for i in range(num_batches):

        model.zero_grad()
        # TODO:see notes
        X_local_minibatch, y_local_minibatch = (
            train_X[i * batch_size : (i + 1) * batch_size,],
            train_Y[i * batch_size : (i + 1) * batch_size,],
        )

        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)
        y_local_minibatch = torch.max(y_local_minibatch, 1)[1]

        y_pred = model(X_local_minibatch)                # fwd the bass (forward pass)
        loss = loss_function(y_pred, y_local_minibatch)  # compute loss
        loss.backward()                                  # reeeeewind (backward pass)
        optimizer.step()                                 # parameter update

        train_running_loss += loss.detach().item()       # unpacks the tensor into a scalar value
        train_acc += model.get_accuracy(y_pred, y_local_minibatch)

    print(
        "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f"
        % (epoch, train_running_loss / num_batches, train_acc / num_batches)
    )

    print("Validation ...")  # should this be done every N epochs
    if epoch % 10 == 0:
        val_running_loss, val_acc = 0.0, 0.0

        # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
        with torch.no_grad():
            model.eval()

            model.hidden = model.init_hidden()
            for i in range(num_dev_batches):
                X_local_validation_minibatch, y_local_validation_minibatch = (
                    dev_X[i * batch_size : (i + 1) * batch_size,],
                    dev_Y[i * batch_size : (i + 1) * batch_size,],
                )
                X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

                y_pred = model(X_local_minibatch)
                val_loss = loss_function(y_pred, y_local_minibatch)

                val_running_loss += (
                    val_loss.detach().item()
                )  # unpacks the tensor into a scalar value
                val_acc += model.get_accuracy(y_pred, y_local_minibatch)

            model.train()  # reset to train mode after iterationg through validation data
            print(
                "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f | Val Loss %.4f  | Val Accuracy: %.2f"
                % (
                    epoch,
                    train_running_loss / num_batches,
                    train_acc / num_batches,
                    val_running_loss / num_dev_batches,
                    val_acc / num_dev_batches,
                )
            )

        epoch_list.append(epoch)
        val_accuracy_list.append(val_acc / num_dev_batches)
        val_loss_list.append(val_running_loss / num_dev_batches)
