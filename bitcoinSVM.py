import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as r2
import matplotlib.pyplot as plt

data = np.genfromtxt("bitcoindata.csv", delimiter=",")
X = np.copy(data[:, 1])
# X = X.reshape(-1, 1)
y = np.copy(data[:, 0])
# y = y.reshape(-1, 1)



# df = pd.read_excel('bitcoindata.xlsx')

# 70% training and 30% test chosen randomly with seed=106
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=106)


# Reshaping
X_testes = X_test.reshape(-1, 1)
X_traines = X_train.reshape(-1, 1)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# X_shape = X.reshape(-1, 1)


# Scaling the X data between -1 and 1
# scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_traines)
scaling = StandardScaler().fit(X_traines)
X_traines = scaling.transform(X_traines)
X_testes = scaling.transform(X_testes)
# X_shape_scale = scaling.transform(X_shape)

# Debugging
# print(X_train.shape, X_test.dtype, type(y_train), type(y_test))
# print(df.head())
# print(data.shape)
# print(X.dtype, y.dtype)


# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X_traines, y_train)
y_rbf_pred = y_rbf.predict(X_testes)
y_lin = svr_lin.fit(X_traines, y_train)
y_lin_pred = y_lin.predict(X_testes)
y_poly = svr_poly.fit(X_traines, y_train)
y_poly_pred = y_poly.predict(X_testes)

# #############################################################################
# Look at the results

# Debugging
# print("y_test:", y_test.shape)
# print("X_testes:", X_testes.ravel().shape)
# print(y_rbf_pred.shape, y_lin_pred.shape, y_poly_pred.shape)

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
rmse_lin = r2(y_test, y_lin_pred)
rmse_poly = r2(y_test, y_poly_pred)

print("rmse_rbf: ", rmse_rbf)
print("rmse_lin: ", rmse_lin)
print("rmse_poly: ", rmse_poly)
