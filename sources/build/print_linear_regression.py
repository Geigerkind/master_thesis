import numpy as np
from sklearn.linear_model import LinearRegression

distinct_number_of_locations = np.asarray([9, 16, 17, 25, 32, 48, 51, 102])
dt_pb_5 = 100 * np.asarray([0.9862, 0.9596, 0.9871, 0.9787, 0.9395, 0.9042, 0.9208, 0.8735])
knn_pb_5 = 100 * np.asarray([0.9776, 0.9208, 0.9724, 0.9605, 0.8451, 0.8318, 0.8876, 0.7472])
dt_pb_5_cont = 100 * np.asarray([0.9824, 0.8949, 0.9864, 0.9639, 0.8834, 0.8932, 0.896, 0.8777])
knn_pb_5_cont = 100 * np.asarray([0.9617, 0.7177, 0.9333, 0.8411, 0.6524, 0.4508, 0.7539, 0.4882])

X = distinct_number_of_locations[:, np.newaxis]

reg = LinearRegression().fit(X, dt_pb_5)
print(reg.coef_)

reg = LinearRegression().fit(X, knn_pb_5)
print(reg.coef_)

reg = LinearRegression().fit(X, dt_pb_5_cont)
print(reg.coef_)

reg = LinearRegression().fit(X, knn_pb_5_cont)
print(reg.coef_)
