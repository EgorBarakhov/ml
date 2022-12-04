import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy import stats


def dist(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def get_neighbours(test_row, X_train, y_train, k):
    distances = list()

    for (train_row, train_class) in zip(X_train, y_train):
        _dist = dist(train_row, test_row)
        distances.append((_dist, train_class))

    distances.sort(key=lambda x: x[0])

    neighbours = list()
    for i in range(k):
        neighbours.append(distances[i][1])

    return neighbours


def predict(X_test, X_train, y_train, k):
    preds = []
    for test_row in X_test:
        nearest_neighbours = get_neighbours(test_row, X_train, y_train, k)
        majority = stats.mode(nearest_neighbours)[0][0]
        preds.append(majority)
    return np.array(preds)


def accuracy(preds, y_test):
    return 100 * (preds == y_test).mean()


if __name__ == '__main__':
    data = pd.read_csv('data.csv', index_col='id').reset_index(drop=True)
    data.drop('Unnamed: 32', axis=1, inplace=True)

    print('Dataframe shape:', data.shape)
    print(data.head(3))

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    y = (y == 'M').astype('int')
    print(X.head(3))
    print(y.head(3))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4)

    for k in range(6):
        if k == 0:
            continue
        preds = predict(X_test.values, X_train.values, y_train, k)
        print(f'K: {k}, accuracy: {accuracy(preds, y_test):.3f} %')
