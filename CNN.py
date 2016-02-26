import numpy as np
import pandas as pd
import theano
import argparse
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from scipy.stats import itemfreq
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Flatten, Reshape
from keras.optimizers import Adam
pd.options.mode.chained_assignment = None  # default='warn'


def mean_absolute_percentage_error(y_true, y_pred):

    """
    Note: does not handle mix 1d representation
    if _is_1d(y_true):
        y_true, y_pred = _check_1d_array(y_true, y_pred)
    """

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def fill_missing_time(row):

    """
    Present time-wise interpolation for one single row, with missing values.
    """

    row.index = pd.to_datetime(row.index)
    return row.interpolate(method="time").fillna(method="backfill")

# Loading
# --------

# Load data. Merge train with output.

train = pd.read_csv('Data/training_input.csv', delimiter=',')
output = pd.read_csv('Data/training_output.csv', delimiter=';')
test = pd.read_csv('Data/testing_input.csv', delimiter=',')
train = pd.merge(train, output, on='ID', how='inner')

# Drop the '09:30:00' column.

train.drop(["09:30:00"], axis=1, inplace=True)
test.drop(["09:30:00"], axis=1, inplace=True)

# Separate train data with no missing entries from the rest.

train_filled = train.drop(pd.isnull(train).any(1).nonzero()[0]).reset_index(drop=True)
train_missing = train[~train["ID"].isin(train_filled["ID"].tolist())]
train_full = train_filled

print "Dataset shape after filling :\n", train_full.shape

# Split data.

features = train_full.drop(["ID", "date", "product_id", "TARGET"], axis=1)
x, y = features.values, train_full['TARGET'].values

x_train, x_val, y_train, y_val = train_test_split(
    x[:50000], y[:50000], test_size=0.2, random_state=42)

# Reshape data.

train_shape = (x_train.shape[0], x_train.shape[1], 1)
val_shape = (x_val.shape[0], x_val.shape[1], 1)

x_train_2 = np.reshape(x_train, train_shape).astype(theano.config.floatX)
x_val_2 = np.reshape(x_val, val_shape).astype(theano.config.floatX)

# Training
# --------

# Convolution Model.

model = Sequential()
model.add(Convolution1D(nb_filter=16,
                        filter_length=2,
                        init='glorot_uniform',
                        input_shape=(x_train.shape[1], 1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Convolution1D(nb_filter=32,
                        filter_length=4,
                        init='glorot_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

ap = argparse.ArgumentParser()
ap.add_argument('-lr', default=1e-3, type=float)
ap.add_argument('-bs', default=50, type=int)
ap.add_argument('-ne', default=10, type=int)
opts = ap.parse_args()

adam = Adam(lr=opts.lr)
model.compile(loss='mean_absolute_percentage_error', optimizer=adam)

print 'Training | Batch size :', opts.bs, ", Number of epochs :", opts.ne
model.fit(x_train_2, y_train, batch_size=opts.bs, nb_epoch=opts.ne,
          validation_data=(x_val_2, y_val), show_accuracy=True)
score, acc = model.evaluate(x_val_2, y_val, batch_size=opts.bs,
                            show_accuracy=True)

print 'Test score :', score
print 'Test accuracy:', acc