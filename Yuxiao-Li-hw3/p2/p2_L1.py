from p2 import *


def cal_loss_l1(pred, Y_onehot, theta, lam):
    m = len(pred)
    loss = np.sum(-np.log(pred[Y_onehot == 1])) / Y_onehot.shape[0]
    loss += (lam / m) * np.linalg.norm(theta, ord=1)
    return loss


def grad1(X, theta, Y_onehot, lam):
    m = Y_onehot.shape[0]
    pred = softmax(X @ theta)
    dl = np.where(Y_onehot == 1, 1 - pred, 0)
    gd = -X.T.dot(dl) / Y_onehot.shape[0]
    gd += lam / m * np.sign(theta)
    return gd


def prox_gd1(X, y, theta, lam, epoch):
    # calculate gradient
    mk = 0.1 / np.sqrt(epoch + 1)  # diminishing step
    a = theta - mk * grad1(X, theta, y, lam)
    # gradient descent update
    return prox(a, lam * mk)


iteration = 10000  # for submission, it is suggested to set epochs=10000
lam_list = [0.001, 0.01, 0.1, 1, 100]

# initialization
Theta = np.zeros(shape=(feature_dim, num_class))
train_logs = {}
for lamda in lam_list:
    train_log = []
    Theta = np.zeros_like(Theta)

    for i in range(iteration):
        # train loss
        pred_train = softmax(sigmoid(np.matmul(X_train, Theta)))  # (n_train, num_class)
        train_loss = cal_loss_l1(pred_train, Y_train_onehot, Theta, lamda)

        # test loss
        pred_test = softmax(sigmoid(np.matmul(X_test, Theta)))  # (n_test, num_class)
        test_loss = cal_loss_l1(pred_test, Y_test_onehot, Theta, lamda)

        # test accuracy
        test_acc = np.sum(pred_test.argmax(axis=1) == Y_test) / test_sz

        train_log.append((train_loss, test_loss, test_acc))

        # Proximal gradient descent
        Theta = prox_gd1(X_train, Y_train_onehot, Theta, lamda, i)

    train_logs[lamda] = list(zip(*train_log))

t = np.arange(iteration)

# plot train loss comparison
plt.figure(0)
for lamda in lam_list:
    plt.plot(t, train_logs[lamda][0], label=lamda)
plt.title('Training Loss with L1 regulation')
plt.xlabel('Epoch')
plt.legend()

# plot test loss comparison
plt.figure(1)
for lamda in lam_list:
    plt.plot(t, train_logs[lamda][1], label=lamda)
plt.title('Test Loss with L1 regulation')
plt.xlabel('Epoch')
plt.legend()

# plot accuracy
plt.figure(2)
for lamda in lam_list:
    plt.plot(t, train_logs[lamda][2], label=lamda)
plt.title('Test Accuracy with L1 regulation')
plt.xlabel('Epoch')
plt.legend()

plt.show()
