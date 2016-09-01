import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l2, activity_l2
from keras.utils.visualize_util import plot

from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)

dataframe = pd.read_csv('data/housing.csv', delim_whitespace=True, header=None)

threshold = (dataframe.shape[0] * 9) // 10

train_dataframe = dataframe[0: threshold]
test_dataframe = dataframe[threshold:]

train = train_dataframe.values
test = test_dataframe.values

# split into input (X) and output (Y) variables
X_train = train[:, 0:13]
Y_train = train[:, 13]

X_test = test[:, 0:13]
Y_test = test[:, 13]

model = Sequential()

model.add(Dense(output_dim=13, input_dim=13, activation='relu', W_regularizer=l2(0.01)))
model.add(Dense(output_dim=13, input_dim=13, activation='relu', W_regularizer=l2(0.01)))
model.add(Dense(output_dim=1, activation='relu'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train, nb_epoch=200, batch_size=20, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)

predict = pd.DataFrame(model.predict_proba(X_test), columns=['predict'])
fact = pd.DataFrame(Y_test, columns=['fact'])

print(pd.concat([fact, predict], axis=1))

print('')
print(score)
plot(model, to_file='model.png', show_shapes=True)

SVG(model_to_dot(model).create(prog='dot', format='svg'))
