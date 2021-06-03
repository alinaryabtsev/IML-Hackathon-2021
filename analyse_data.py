import pandas as pd
import numpy as np
import sys

from pandas.core.frame import DataFrame




#work with: release date, Belongs to collection, Budget, genres
# release date: split into months (10 days? clustering?)
# Belongs to collection split into: 
#                                   1. Is it included in a collection?
#                                   2. the collection?
# budget : delete Unreasonable data / make them n/a
# genres:  every genera is a new feature 



def get_X_y(file_name):
    df = pd.read_csv(file_name)
    # print(df.head())

    col1 = "vote_average"
    col2 = "revenue"

    y_train = df[[col1,col2]]
    X_train = df.drop([col1,col2],axis=1)

    return X_train, y_train

def analyse_releaseDate(X,y):

    df['date'] = pd.to_datetime(df)
    # print(X[['release_date']])
    df_date['date'] = pd.to_datetime(X['release_date'], errors='coerce',dayfirst=True)
    print(df_date)
    print("****")
    print(df_date['release_date'].dt.year)
    df_date['year']= df_date['release_date'].dt.year
    df_date['month']= df_date['release_date'].dt.month
    df_date['day']= df_date['release_date'].dt.day
    print(df_date)

if __name__ == '__main__':
    if len(sys.argv) >1:
        file_name = sys.argv[1]
    else:
        file_name = 'movies_dataset.csv'
    X, y = get_X_y(file_name)
    analyse_releaseDate(X,y)
    





