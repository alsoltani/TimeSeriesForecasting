import numpy as np
import pandas as pd
import pywt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV


# Wavelet transform class.

class DictT(object):

    def __init__(self, name, level):
            self.name = name
            self.level = level
            self.sizes = []

    def dot(self, mat):

        m = []

        if mat.shape[0] != mat.size:
            for i in xrange(mat.shape[1]):
                c = pywt.wavedec(mat[:, i], self.name, level=self.level)
                self.sizes.append(map(len, c))
                c = np.concatenate(c)
                m.append(c)
            return np.asarray(m).T
        else:
            c = pywt.wavedec(mat, self.name, level=self.level)
            self.sizes.append(map(len, c))
            return np.concatenate(c)


# Mean Absolute Percentage Error.

def mean_absolute_percentage_error(y_true, y_pred):

    """
    Note: does not handle mix 1d representation
    if _is_1d(y_true):
        y_true, y_pred = _check_1d_array(y_true, y_pred)
    """

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":

    # Import data.

    training = pd.read_csv('Data/training_input.csv', delimiter=',')
    output = pd.read_csv('Data/training_output.csv', delimiter=';')

    training = training.drop(pd.isnull(training).any(1).nonzero()[0]).reset_index(drop=True)
    training = pd.merge(training, output, on='ID', how='inner')

    x = training.drop(["ID", "date", "product_id", "TARGET"], axis=1).values
    y = training["TARGET"].values

    print "Full labelled set :\n", x.shape, "\n", y.shape

    # Wavelet dictionary.

    wave_name = 'db17'
    wave_level = None
    wavelet_operator_t = DictT(level=wave_level, name=wave_name)

    basis_t = wavelet_operator_t.dot(np.identity(x.shape[1]))
    basis_t /= np.sqrt(np.sum(basis_t ** 2, axis=0))
    basis = basis_t.T

    # Regression.

    x_train, x_val, y_train, y_val = train_test_split(
        x.dot(basis), y, test_size=0.2, random_state=0)

    print "Training set :\n", x_train.shape, "\n", y_train.shape
    print "Validation set :\n", x_val.shape, "\n", y_val.shape

    # Grid search.

    reg_grid = {
        "max_depth": [10, 20, 40, None],
        "max_features": [20, 30, 40, 'auto'],
        "min_samples_split": [1, 5, 10],
        "min_samples_leaf": [1, 5, 10],
        "bootstrap": [True, False]}

    reg = GridSearchCV(RandomForestRegressor(n_estimators=50, n_jobs=-1),
                       param_grid=reg_grid, n_jobs=-1, verbose=5)
    reg.fit(x_train, y_train)
    y_val_pred = reg.predict(x_val)

    params = reg.best_params_
    print "Best params:\n", params

    # MAPE.

    print "MAPE :", mean_absolute_percentage_error(y_val, y_val_pred)
