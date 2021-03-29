import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import config
from LSTM_with_Attention import LSTM_with_Attention
from utils import dataloader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = config.DATA_DIR
train_X, train_Y, dev_X, dev_Y = dataloader.load_data(DATA_DIR)

batch_size = config.BATCH_SIZE
num_epochs = config.NUM_EPOCHS
initial_lr = config.LR
hidden_size = config.HIDDEN_SIZE
num_layers = config.NUM_LAYERS

# Define model
print("Build LSTM model ..")
model = LSTM_with_Attention(
    input_size=6,  # TODO : 6
    hidden_size=hidden_size,
    batch_size=batch_size,
    output_size=2,  # TODO : 2
    num_layers=num_layers
)
model.to(device)
loss_function = nn.NLLLoss()
val_acc = 0.0
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 120, 160, 200], gamma=0.8)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("\n Training on GPU")
else:
    print("\n No GPU, training on CPU")

num_batches = int(train_X.shape[0] / batch_size)
num_dev_batches = int(dev_X.shape[0] / batch_size)

val_loss_list, val_accuracy_list, epoch_list = [], [], []

print("Training ...")
print("learning rate: ", optimizer.defaults['lr'])
for epoch in range(num_epochs):

    train_running_loss, train_acc = 0.0, 0.0
    # print(model.hidden)
    # model.hidden = model.init_hidden()
    for i in range(num_batches):

        model.zero_grad()
        # TODO:see notes
        X_local_minibatch, y_local_minibatch = (
            train_X[i * batch_size: (i + 1) * batch_size, ],
            train_Y[i * batch_size: (i + 1) * batch_size, ],
        )
        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)
        y_local_minibatch = torch.max(y_local_minibatch, 1)[1] # 返回每行最大值(gt)的索引

        y_pred = model(X_local_minibatch)                # fwd the bass (forward pass)
        loss = loss_function(y_pred, y_local_minibatch)  # compute loss
        loss.backward()                                  # reeeeewind (backward pass)
        optimizer.step()                                 # parameter update
        train_running_loss += loss.detach().item()       # unpacks the tensor into a scalar value
        train_acc += model.get_accuracy(y_pred, y_local_minibatch)

    print("learning rate: ", optimizer.param_groups[0]['lr'])
    scheduler.step()
    print(
        "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f"
        % (epoch, train_running_loss / num_batches, train_acc / num_batches)
    )

    print("Validation ...")  # should this be done every N epochs
    if (epoch + 1) % 5 == 0:
        val_running_loss, val_acc = 0.0, 0.0

        # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
        with torch.no_grad():
            model.eval()

            # model.hidden = model.init_hidden()
            for i in range(num_dev_batches):
                X_local_validation_minibatch, y_local_validation_minibatch = (
                    dev_X[i * batch_size: (i + 1) * batch_size, ],
                    dev_Y[i * batch_size: (i + 1) * batch_size, ],
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

# torch.save(model, "./model/tt")
plt.plot(epoch_list, val_loss_list)
plt.xlabel("# of epochs")
plt.ylabel("Loss")
plt.title("LSTM: Loss vs # epochs")
plt.savefig('graph.png')
plt.show()

plt.plot(epoch_list, val_accuracy_list, color="red")
plt.xlabel("# of epochs")
plt.ylabel("Accuracy")
plt.title("LSTM: Accuracy vs # epochs")
plt.savefig('graph_1.png')
plt.show()

# print("max val accuracy: ", max(val_acc))
# torch.save(model, './model/test.pkl')
