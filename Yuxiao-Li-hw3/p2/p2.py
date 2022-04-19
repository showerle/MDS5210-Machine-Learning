#! python3

from random import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(666)


def load_dataset(file):
    """load csv dataset"""
    df_train = pd.read_csv(file)
    X = df_train.iloc[:, :-1]
    Y = df_train.iloc[:, -1].astype(int)
    if Y.min() == 1:
        Y -= 1  # minus 1 so that label starts from 0
    X = (X - X.mean()) / X.std()  # normalization
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()

    return (X_train, Y_train), (X_test, Y_test)


def sigmoid(x):
    exp_x = np.exp(-x)
    g = 1 / (1 + exp_x)
    return g


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cal_loss(pred, Y_onehot):
    """ calculate multinomial logistic regression loss (regularization term NOT included)
    pred: (sample_num, class_num)
    Y_onehot: (sample_num, class_num)
    """
    loss = np.sum(-np.log(pred[Y_onehot == 1])) / Y_onehot.shape[0]
    loss = np.squeeze(loss)  # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (loss.shape == ())
    return loss  # TODO·9·1·11·20


def gradient(X, theta, Y_onehot):
    """
        Y: onehot encoded
        """
    pred = softmax(X @ theta)
    dl = np.where(Y_onehot == 1, 1 - pred, 0)
    gd = -X.T.dot(dl) / Y_onehot.shape[0]
    return gd


def sto_gradient(Xi, theta, yi):
    # calculate gradient
    pred = softmax(Xi @ theta)
    pred = np.where(yi == 1, 1 - pred, 0)
    return -Xi[:, np.newaxis].dot(pred[np.newaxis, :])


def prox(a, t):
    # gradient descent update
    theta_update = np.zeros(a.shape)
    theta_update[a > t] = a[a > t] - t
    theta_update[a < -t] = a[a < -t] + t
    return theta_update


def prox2(a, lam, mu):
    # gradient descent update
    t = lam * mu
    theta_update = np.zeros(a.shape)
    theta_update[a > t] = a[a > t] - lam
    theta_update[a < -t] = a[a < -t] + lam
    return theta_update


def prox_gd(X, y, theta, lam, epoch):
    # calculate gradient
    mk = 0.1 / np.sqrt(epoch + 1)  # diminishing step
    a = theta - mk * gradient(X, theta, y)
    # gradient descent update
    return prox(a, lam * mk)


def prox_sgd(X, y, theta, epoch):
    global mk, thetak
    identity_k = np.identity(num_class)
    batch = list(range(train_sz))
    shuffle(batch)

    # # decaying step sise
    if epoch < 3000:
        mk = 0.005 / np.sqrt(epoch + 1)  # diminishing step
    elif 3000 <= epoch <= 5000:
        mk = 0.005 / np.sqrt(epoch + 1)  # diminishing step
    elif epoch > 5000:
        mk = 0.0025 / np.sqrt(epoch + 1)  # diminishing step

    # # RR-SGD
    for num in batch:
        a = theta - mk * sto_gradient(X[num], theta, y[num])  # gradient descent update
        thetak = prox2(a, lam, mu)
        theta = thetak
    return theta


def prox_momentum_sgd(X, y, theta0, theta1, epoch):
    # RR-SGD
    global mk
    batch = list(range(train_sz))
    shuffle(batch)
    # decaying step size
    if epoch < 3000:
        mk = 0.01 / np.sqrt(epoch + 1)  # diminishing step
    elif 3000 <= epoch <= 5000:
        mk = 0.005 / np.sqrt(epoch + 1)  # diminishing step
    elif epoch > 5000:
        mk = 0.0025 / np.sqrt(epoch + 1)  # diminishing

    for num in batch:
        wk = theta1 + epoch / (epoch + 3) * (theta1 - theta0)  # calculate momentum
        a = wk - mk * sto_gradient(X[num], wk, y[num])
        theta2 = prox2(a, lam, mu)

        theta0 = theta1
        theta1 = theta2

    return theta1, theta0


def prox_agd(X, y, theta0, theta1, epoch):
    wk = theta1 + epoch / (epoch + 2) * (theta1 - theta0)  # calculate momentum
    a = wk - mu * gradient(X, wk, y)
    thetak = prox2(a, lam, mu)
    theta0 = theta1
    theta1 = thetak
    return theta1, theta0


# load data
(X_train, Y_train), (X_test, Y_test) = load_dataset('fetal_health.csv')
train_sz, test_sz = X_train.shape[0], X_test.shape[0]
feature_dim = X_train.shape[1]
num_class = np.max(Y_train) + 1  # y_min starts from 0
Y_train_onehot = np.eye(num_class)[Y_train]  # (n_train, num_class)
Y_test_onehot = np.eye(num_class)[Y_test]  # (n_test, num_class)

# hypermarkets
epochs = 1000  # for submission, it is suggested to set epochs=10000
lam = 0.001
mu = 10e-3

# initialization
Theta = np.zeros(shape=(feature_dim, num_class))

# optimization algorithms, you may test it one by one
train_modes = ['prox_gd', 'prox_sgd', 'prox_momentum_sgd', 'prox_agd']

train_logs = {}
# for train_mode in train_modes:
#     train_log = []
#     Theta = np.zeros_like(Theta)
#
#     # momentum preparation
#     Theta0 = Theta
#     Theta1 = Theta0 - mu * gradient(X_train, Theta0, Y_train_onehot)
#
#     for epoch in range(epochs):
#         # train loss
#         pred_train = softmax(np.matmul(X_train, Theta))  # (n_train, num_class)
#         train_loss = cal_loss(pred_train, Y_train_onehot)
#
#         # test loss
#         pred_test = softmax(np.matmul(X_test, Theta))  # (n_test, num_class)
#         test_loss = cal_loss(pred_test, Y_test_onehot)
#
#         # test accuracy
#         test_acc = np.sum(pred_test.argmax(axis=1) == Y_test) / test_sz
#
#         train_log.append((train_loss, test_loss, test_acc))
#         # print(f"epoch:{epoch}, train_loss:{train_loss}, test_loss:{test_loss}, test_acc:{test_acc}")
#
#         # Proximal gradient descent
#         if train_mode == 'prox_gd':
#             Theta = prox_gd(X_train, Y_train_onehot, Theta, lam, epoch)
#
#         # Proximal stochastic gradient descent
#         elif train_mode == 'prox_sgd':
#             Theta = prox_sgd(X_train, Y_train_onehot, Theta, epoch)
#
#         elif train_mode == 'prox_momentum_sgd':
#             Theta1, Theta0 = prox_momentum_sgd(X_train, Y_train_onehot, Theta0, Theta1, epoch)
#             Theta = Theta1
#
#         # Accelerated gradient descent
#         elif train_mode == 'prox_agd':
#             Theta1, Theta0 = prox_agd(X_train, Y_train_onehot, Theta0, Theta1, epoch)
#             Theta = Theta1
#
#     train_logs[train_mode] = list(zip(*train_log))
#
# t = np.arange(epochs)
#
# # plot train loss comparison
# plt.figure(0)
# for train_mode in train_modes:
#     plt.plot(t, train_logs[train_mode][0], label=train_mode)
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.legend()
#
# # plot test loss comparison
# plt.figure(1)
# for train_mode in train_modes:
#     plt.plot(t, train_logs[train_mode][1], label=train_mode)
# plt.title('Test Loss')
# plt.xlabel('Epoch')
# plt.legend()
#
# # plot accuracy
# plt.figure(2)
# for train_mode in train_modes:
#     plt.plot(t, train_logs[train_mode][2], label=train_mode)
# plt.title('Test Accuracy')
# plt.xlabel('Epoch')
# plt.legend()
#
# plt.show()
