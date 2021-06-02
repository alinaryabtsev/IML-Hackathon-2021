from io import RawIOBase
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

TRAIN_RATIO=0.8
VALIDATION_RATIO=0.1
TEST_RATIO=0.1


def split_data(file_name):
    df = pd.read_csv(file_name)
    print(df.head())

    col1 = "vote_average"
    col2 = "revenue"

    y = df[[col1,col2]]
    # print(y.head())
    X = df.drop([col1,col2],axis=1)
    # print(X.head())
    # print(X.columns)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X,y, test_size= TEST_RATIO+VALIDATION_RATIO, random_state = 5)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp,y_tmp, test_size= 0.5, random_state = 5)
    print(X.shape[0])
    print(X_train.shape[0])
    print(X_test.shape[0])
    print(X_val.shape[0])

    df_train = pd.concat([X_train, y_train], axis=1)
    df_validation = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_train.to_csv("train.csv")
    df_validation.to_csv("validation.csv")
    df_test.to_csv("test.csv")






if __name__ == '__main__':
    if len(sys.argv) >1:
        file_name = sys.argv[1]
    else:
        file_name = 'movies_dataset.csv'
    split_data(file_name)
    










