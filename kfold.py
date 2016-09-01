import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, LearningRateScheduler


def load_data(filename):
    dataframe = pd.read_csv(filename, delim_whitespace=True, header=None)
    data = dataframe.values
    x = data[:, 0:13]
    y = data[:, 13]

    return x, y


def create_model():
    model = Sequential()
    model.add(Dense(output_dim=13, input_dim=13, activation='relu', W_regularizer=l2(0.01)))
    model.add(Dense(output_dim=1, activation='relu'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def train_and_evaluate_model(model, train_data, train_labels, test_data, test_labels):
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit(train_data, train_labels, nb_epoch=1000, batch_size=20, validation_data=(test_data, test_labels),
              verbose=False, callbacks=[early_stopping])

    return model.evaluate(test_data, test_labels)

n_folds = 5
X, y = load_data('data/housing.csv')
skf = KFold(len(y), n_folds=n_folds, shuffle=True)

scores = []

for i, (train, test) in enumerate(skf):
        print("\nRunning Fold", i+1, "/", n_folds)
        model = create_model()
        score = train_and_evaluate_model(model, X[train], y[train], X[test], y[test])
        scores.append(score)

scores = np.asarray(scores)
print("\n", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
