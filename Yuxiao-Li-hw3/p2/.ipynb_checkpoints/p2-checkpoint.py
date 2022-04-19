import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(666)

def load_dataset(file):
    """load csv dataset"""
    df_train = pd.read_csv(file)
    X = df_train.iloc[:,:-1]
    Y = df_train.iloc[:,-1].astype(int)
    if Y.min() == 1:
        Y -= 1 # minus 1 so that label starts from 0
    X = (X - X.mean()) / X.std() # normalization
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=10)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy() 
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()
    
    return (X_train, Y_train), (X_test, Y_test)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cal_loss(pred, Y_onehot):
    """ calculate multinomial logistic regression loss (regularization term NOT included)
    pred: (sample_num, class_num)
    Y_onehot: (sample_num, class_num)
    """
    return 0 # TODO

# load data
(X_train, Y_train), (X_test, Y_test) = load_dataset('fetal_health.csv')
train_sz, test_sz = X_train.shape[0], X_test.shape[0]
feature_dim = X_train.shape[1]
num_class = np.max(Y_train) + 1 # y_min starts from 0
Y_train_onehot = np.eye(num_class)[Y_train] # (n_train, num_class)
Y_test_onehot = np.eye(num_class)[Y_test] # (n_test, num_class)

# hyperparameters
epochs = 100 # for submission, it is suggested to set epochs=10000

# initialization
Theta = np.zeros(shape=(feature_dim, num_class)) 

# optimization algorithms, you may test it one by one
train_modes = ['prox_gd', 'prox_sgd', 'prox_momentum_sgd', 'prox_agd']

train_logs = {}
for train_mode in train_modes:
    train_log = []
    Theta = np.zeros_like(Theta)
    for epoch in range(epochs):
        # train loss
        pred_train = softmax(np.matmul(X_train, Theta)) # (n_train, num_class)
        train_loss = cal_loss(pred_train, Y_train_onehot)

        # test loss
        pred_test = softmax(np.matmul(X_test, Theta)) # (n_test, num_class)
        test_loss = cal_loss(pred_test, Y_test_onehot)

        # test accuracy
        test_acc = np.sum(pred_test.argmax(axis=1)==Y_test) / test_sz

        train_log.append((train_loss, test_loss, test_acc))
        print(f"epoch:{epoch}, train_loss:{train_loss}, test_loss:{test_loss}, test_acc:{test_acc}")

        # Proximal gradient descent
        if train_mode == 'prox_gd':
            pass # TODO

        # Proximal stochastic gradient descent
        elif train_mode == 'prox_sgd':
            pass # TODO
        
        elif train_mode == 'prox_momentum_sgd':
            pass # TODO

        # Accelerated gradient descent
        elif train_mode == 'prox_agd':
            pass # TODO

    train_logs[train_mode] = list(zip(*train_log))

t = np.arange(epochs)

# plot train loss comparison
plt.figure(0)
for train_mode in train_modes:
    plt.plot(t, train_logs[train_mode][0], label=train_mode)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.legend()

# plot test loss comparison
plt.figure(1)
for train_mode in train_modes:
    plt.plot(t, train_logs[train_mode][1], label=train_mode)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.legend()

# plot accuracy
plt.figure(2)
for train_mode in train_modes:
    plt.plot(t, train_logs[train_mode][2], label=train_mode)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()