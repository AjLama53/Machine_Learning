# This file is for the processing of the data
# load_data is to load the data
# scale_data is to scale the data


import pandas as pd
from sklearn.preprocessing import StandardScaler



def load_data(file):
    df = pd.read_csv(file)

    X = df.drop(['Ensembl_ID', 'Class'], axis=1)
    Y = df['Class']

    labels = sorted(Y.unique())

    return X, Y, labels


def scale_data(X):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, X.columns)

    return X_scaled


