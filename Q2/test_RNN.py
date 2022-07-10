import torch
from train_RNN import RNN, to_tensor
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = torch.load("./models/20433010_RNN_model.torch")
    model.eval()
    with open("./models/scalar.pk", "rb") as f:
        scalars = pickle.load(f)

    test_df = pd.read_csv("./data/test_data_RNN.csv", header=None)
    test_df[test_df.columns[0]] = pd.to_datetime(test_df[test_df.columns[0]])
    test_data = test_df.sort_values(by=[test_df.columns[0]])
    test_data = test_data[test_data.columns[1:]]
    # label before normalization
    y_test_raw = np.array(test_data[test_data.columns[-1]].values)

    for col in test_data.columns:
        test_data[col] = scalars[col].transform(test_data[col].values.reshape(-1, 1))

    scaled_test_data = test_data.values

    x_test = to_tensor(scaled_test_data[:, :-1].reshape(scaled_test_data.shape[0], 1, -1))
    y_test = scaled_test_data[:, -1:]

    y_test_pred = model(x_test)

    mse = torch.nn.MSELoss(reduction='mean')
    loss = mse(y_test_pred, to_tensor(y_test))
    print("loss after scaling", loss.detach().numpy())

    # mse calculated with reverse normalized predictions and labels
    target_scalar = scalars[test_df.columns[-1]]
    y_test_pred_reversed = target_scalar.inverse_transform(y_test_pred.detach().numpy())

    loss = mse(to_tensor(y_test_pred_reversed), to_tensor(y_test_raw.reshape(-1, 1)))
    print("loss before scaling", loss.detach().numpy())

    test_pred = pd.DataFrame({"y_test_pred": y_test_pred_reversed.reshape(-1), "y_test": y_test_raw.reshape(-1)})

    # save predictions and ground truth for test set
    test_pred.to_csv("./data/test_pred.csv", index=False)

    fig = plt.figure()

    ax = sns.lineplot(x=test_df.index, y=y_test_raw.reshape(-1), label="Ground truth", color='grey')
    ax = sns.lineplot(x=test_df.index, y=y_test_pred_reversed.reshape(-1), label="RNN prediction", color='red')
    ax.set_title('Test set prediction', size=10, fontweight='bold')
    ax.set_xlabel("Days", size=15)
    ax.set_ylabel("$", size=15)
    ax.set_xticklabels('', size=10)

    plt.show()
