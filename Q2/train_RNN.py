import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch.nn as nn
import pickle


def split_data(df, feat_names, target_name, window):
    data = df[feat_names]
    date_series = df["Date"]
    data[f"{target_name}_target"] = df[target_name]
    data = data.to_numpy()

    windows = []
    for i in range(len(data) - window):
        windows.append(data[i: i + window, :-1].reshape(1, -1))

    windows = np.array(windows)
    dt = date_series.values[:len(data) - window].reshape(-1, 1)[1:, :]
    x = windows[1:, :]
    y = data[:len(data) - window - 1, -1]

    xy = np.concatenate([dt, x.reshape(x.shape[0], -1), y.reshape(-1, 1)], axis=1)

    np.random.seed(123)
    np.random.shuffle(xy)

    split = 0.7
    train_set_size = int(xy.shape[0] * split)
    dt_train = xy[:train_set_size, :1]
    x_train = xy[:train_set_size, 1:-1].reshape(train_set_size, 1, -1)
    y_train = xy[:train_set_size, -1:]

    dt_test = xy[train_set_size:, :1]
    x_test = xy[train_set_size:, 1:-1].reshape(xy.shape[0] - train_set_size, 1, -1)
    y_test = xy[train_set_size:, -1:]
    return dt_train, x_train, y_train, dt_test, x_test, y_test


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, scalars):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.scalars = scalars

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


def to_tensor(np_array):
    return torch.from_numpy(np_array).type(torch.Tensor)


if __name__ == "__main__":
    # df = pd.read_csv("./q2_dataset.csv")
    #
    # df.columns = df.columns.str.strip()
    #
    # feats = ['Volume', 'Open', 'High', 'Low']
    # target = 'Open'
    #
    # window = 3
    # dt_train, x_train, y_train, dt_test, x_test, y_test = split_data(df, feats, target, window)
    # train_data = np.concatenate([dt_train, x_train.reshape(x_train.shape[0], -1), y_train], axis=1)
    # test_data = np.concatenate([dt_test, x_test.reshape(x_test.shape[0], -1), y_test], axis=1)
    #
    # np.savetxt("train_data_RNN.csv", train_data, delimiter=",", fmt='%s')
    # np.savetxt("test_data_RNN.csv", test_data, delimiter=",", fmt='%s')

    train_data = pd.read_csv("train_data_RNN.csv", header=None, index_col=None)
    train_data = train_data[train_data.columns[1:]]

    # scale all feature between -1 and 1
    scalers = {}
    for column in train_data.columns:
        scalers[column] = MinMaxScaler(feature_range=(-1, 1))
        train_data[column] = scalers[column].fit_transform(train_data[column].values.reshape(-1, 1))

    x_train = to_tensor(train_data.values[:, :-1].reshape(len(train_data), 1, -1))
    y_train_lstm = to_tensor(train_data.values[:, -1].reshape(-1, 1))

    input_dim = 12
    hidden_dim = 200
    num_layers = 2
    output_dim = 1
    num_epochs = 200

    model = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                scalars=scalers)
    mse = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = mse(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

    print(model)
    torch.save(model, "./models/20433010_RNN_model.torch")
    with open("./models/scalar.pk", "wb") as f:
        pickle.dump(scalers, f)
