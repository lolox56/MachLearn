import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as r2
# import matplotlib.pyplot as plt

data = np.genfromtxt("moreBitcoinch.csv", delimiter=",")
X = np.copy(data[:, 1:])
# X = X.reshape(-1, 1)
y = np.copy(data[:, 0])
# y = y.reshape(-1, 1)

# 70% training and 30% test chosen randomly with seed=106
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=106)



# Reshaping
# X_testes = X_test.reshape(-1, 1)
# X_traines = X_train.reshape(-1, 1)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# X_shape = X.reshape(-1, 1)


# Scaling the X data between -1 and 1
# scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_traines)
scaling = StandardScaler().fit(X_train)
X_traines = scaling.transform(X_train)
X_testes = scaling.transform(X_test)
# X_shape_scale = scaling.transform(X_shape)


# #############################################################################
# Fit regression model
prefactors = list([2.0**i for i in range(-5, 16)])
epsilons = list([2.0**i for i in range(-15, 6)])


max_acc = 0
best = [0, 0, 0]

for prefactor in prefactors:
    for epsilon in epsilons:
        svr_rbf = SVR(kernel='rbf', C=prefactor, gamma=epsilon)
        y_rbf = svr_rbf.fit(X_traines, y_train)
        y_rbf_pred = y_rbf.predict(X_testes)
        accuracy = r2(y_test, y_rbf_pred)
        if accuracy > max_acc:
            max_acc = accuracy
            best[:] = [prefactor, epsilon, max_acc]

# df = pd.DataFrame({'prefactors': prefactors, 'epsilons': epsilons, 'accuracy': accuracy})
print(best)


"""

X_plot = X_testes.ravel()

lw = 2
plt.plot(y_test, color='darkorange', label='test data')
plt.plot(y_rbf_pred, color='navy', lw=lw, label='RBF model')
# plt.plot(y_lin_pred, color='c', lw=lw, label='Linear model')
# plt.plot(y_poly_pred, color='cornflowerblue', lw=lw, label='Polynomial model')
# plt.scatter(X_plot, y_test, color='darkorange', label='data')
# plt.plot(X_plot, y_rbf_pred, color='navy', lw=lw, label='RBF model')
# plt.plot(X_plot, y_lin_pred, color='c', lw=lw, label='Linear model')
# plt.plot(X_plot, y_poly_pred, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

rmse_rbf = r2(y_test, y_rbf_pred)


print("rmse_rbf: ", rmse_rbf)
"""
