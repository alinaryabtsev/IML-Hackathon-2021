import pandas as pd
import numpy as np
import ast
import random
import sys
from pandas.api.types import is_numeric_dtype
from pandas.core.frame import DataFrame


class Preprocessing:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def get_df(self):
        return self.df

    def replace_na_in_tagline(self):
        self.df["tagline"] = self.df["tagline"].fillna("")

    def get_columns_names(self):
        return self.df.columns

    def get_X_y(self, file_name):

        col1 = "vote_average"
        col2 = "revenue"

        y_train = self.df[[col1, col2]]
        X_train = self.df.drop([col1, col2], axis=1)

        return X_train, y_train

    def turn_value_to_nan(self, col_value, op):
        """
        ex:
            turn_value_to_nan({'budget':20})

        :param col_value: dictionaray of columns names and the values that any value lower than that will be replaced with NaN
        """
        for key in col_value.keys():
            a = np.array(self.df[key].values.tolist())
            if op == '<':
                self.df[key] = np.where(a < col_value[key], np.nan, a).tolist()
            else:
                self.df[key] = np.where(a > col_value[key], np.nan, a).tolist()

    def turn_nan_value_to_mean(self, col_names):
        """
        ex:
            turn_value_to_nan({'budget':20})

        :param col_value: dictionaray of columns names and the values that any value lower than that will be replaced with NaN
        """
        for col in col_names:
            m = round(self.get_df()[col].mean())
            self.df[col] = self.df[col].fillna(m)

    def turn_values_to_int(self, col_names, new_type=np.int16):
        """
        ex: 
            turn_values_to_int(['year'],df)
            turn_values_to_int(['month', 'day'],df, new_type=np.int8)

        :param col_names: list of the names of the colums that need to be integers
        """
        for c_name in col_names:
            col = np.array(self.df[c_name], new_type)
            self.df[c_name] = col

    def delete_not_released_movies(self):
        # print(last_year_df[['release_date']].dtypes)
        # last_year_df['today'] = pd.Timestamp('today').strftime("%Y-%m-%d")
        self.get_df()['today'] = pd.to_datetime(
            pd.to_datetime('today').strftime("%Y-%m-%d"))
        self.get_df()['days_till_release'] = (
            self.get_df()['release_date'] - self.get_df()['today']).dt.days
        self.df = self.get_df()[self.get_df()['days_till_release'] < 0]
        self.df = self.df.drop('days_till_release', axis=1)
        self.df = self.df.drop('today', axis=1)

    def preprocess_date(self):
        """
        return the colums added!
        """
        self.df['release_date'] = pd.to_datetime(self.df['release_date'])
        self.df['year'] = self.df['release_date'].dt.year
        self.df['month'] = self.df['release_date'].dt.month
        self.df['day'] = self.df['release_date'].dt.day
        self.turn_value_to_nan({'year': 1}, "<")
        self.turn_value_to_nan({'year': 2021}, ">")
        self.turn_values_to_int(['year'])
        self.turn_values_to_int(['month', 'day'], new_type=np.int8)
        # print("******")
        self.turn_nan_value_to_mean(['year', 'day', 'month'])
        self.df = self.df.drop('release_date', axis=1)
        # print(self.df.columns)
        # return ['year','month', 'day' ]

    def preprocess_budget(self):
        added_cols = self.turn_value_to_nan({'budget': 10}, "<")
        # my_cols = my_cols + added_cols
        self.turn_nan_value_to_mean(['budget'])

    def preprocess_belongs_to_collection(self):
        self.df['belongs_to_collection'] = self.df['belongs_to_collection'].fillna(
            "")
        self.df['binary_collection'] = (
            self.df["belongs_to_collection"] != "").astype(int)
        self.df = self.df.drop('belongs_to_collection', axis=1)

    def preprocess_genres(self):
        self.df['genres'] = self.df.genres.fillna("[]")
        self.convert_json_to_dict("genres")
        dics = [dic for dic in self.df["genres"]]
        
        keys = set()
        for lst in dics:
            for dic in lst:
                keys.add(dic["name"])
        
        for key in keys:
            self.df["genres" + f"_{key}"] = [1 if f"{key}" in string else 0 for string in self.df['genres'].astype('str')]







    def convert_json_to_dict(self, column):
        self.df[column] = self.df[column].apply(
            lambda s: list(ast.literal_eval(s)))
