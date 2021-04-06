
from pandas import read_csv
from datetime import datetime
# load data
#def parse(x):
#	return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv('simulator.ver1.csv', index_col=-1)

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load dataset
dataset = read_csv('simulator.ver1.csv', header=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# split into train and test sets
n = scaled.shape[0]
print(n, type(n), scaled.shape)
train = scaled[:int(n*0.9), :]
test = scaled[int(n*0.9):, :]
# split into input and outputs
train_X, train_y = train[:, :-2], train[:, -2:]
test_X, test_y = test[:, :-2], test[:, -2:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
model.add(LSTM(train_X.shape[2], input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(2))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)

# Graph 1
pyplot.title("Prediction y1")
pyplot.plot(yhat[:, 0], 'red', label='y1 predict rain')
pyplot.plot(test_y[:, 0], 'blue', label='y1 test rain')
pyplot.legend()
pyplot.show()
# Graph 2
pyplot.title("Prediction y2")
pyplot.plot(yhat[:, 1], 'red', label='y2 predict cloud')
pyplot.plot(test_y[:, 1], 'blue', label='y2 test cloud')
pyplot.legend()
pyplot.show()

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, :]), axis=1)
#print(inv_yhat.shape)
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#inv_yhat = inv_yhat.reshape((inv_yhat.shape[0], inv_yhat.shape[2]))
#print(inv_yhat.shape)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 2))
inv_y = concatenate((test_y, test_X[:, :]), axis=1)


inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('***Test RMSE: %.3f***' % rmse)

