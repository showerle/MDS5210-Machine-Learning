from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import time


# plot jth image
def plot_jimage(j):
    plt.title('The jth image is a {label}'.format(label=int(y[j])))
    plt.imshow(X[j].reshape((28, 28)), cmap='gray')
    plt.show()


def train_kernal(X, y, k, parameters, *args):
    start = time.time()
    svm_ = svm.SVC(kernel=k, degree=int(args[0]), random_state=666)
    clf = GridSearchCV(svm_, parameters, scoring="accuracy", cv=fold, n_jobs=-1)
    clf.fit(X, y)
    Pe_1 = 1 - clf.cv_results_["mean_test_score"]

    timecost = time.time() - start
    print(f"train model when \t degree = {args[0]} \t time cost = {timecost:.4f}")

    return clf.best_params_["C"], Pe_1


def test_error(X_test, y_test, k, *args):
    start = time.time()
    svm_ = svm.SVC(kernel=k, degree=int(args[0]), C=args[1], random_state=666)
    svm_.fit(X_test, y_test)
    error = 1 - svm_.score(X_test, y_test)

    timecost = time.time() - start
    print(f"calculate error when \t degree = {args[0]} \t C = {args[1]} \t time cost = {timecost:.4f}")

    return error


def plot_c(C, Pe_1, k, *args):
    plt.figure(1, figsize=(6, 8))
    plt.ylabel("validation error")
    plt.xlabel("C")
    plt.title(f"validation error with respect to C when kernal = {k} degree={args}")
    plt.semilogx(C, Pe_1, marker='o', label="Mean test error with 4-fold CV")
    plt.legend()
    plt.savefig(f'./p3a_{args[0]}')
    plt.show()


def problem1(d):
    bestC, Pe = train_kernal(X_train, y_train, "poly", parameters_a, d)
    best_test_error = test_error(X_test, y_test, "poly", d, bestC)
    print(f"The error with best C is :{best_test_error:.2f} when degree is: {d}")
    plot_c(C_grid, Pe, "poly", d)


fold = 5
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X, y = X.iloc[:70,], y[:70,]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4 / 7, random_state=42)
C_grid = np.logspace(-3, 3, 10)
parameters_a = {"C": C_grid}

if __name__ == '__main__':
    # problem a)
    # threads = [threading.Thread(target=problem1, args=(1,)), threading.Thread(target=problem1, args=(2,))]
    # for t in threads:  # 启动2个线程
    #     t.start()
    # for t in threads:  # 等待线程的结束
    #     t.join()
    for i in [1, 2]:
        problem1(i)
