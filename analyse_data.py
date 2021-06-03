import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json as js
import ast


def preprocess_production_countries():
    countries = df["production_countries"]
    print(countries.head())


def preprocess_runtime():
    df.runtime = df.runtime.where(df.runtime.between(1, 1440))
    sns.barplot(df.runtime, df.revenue) # creating the bar plot
    plt.show()


def preprocess_spoken_languages():
    # print(df.spoken_languages.head())
    pass


def preprocess_homepage():
    homepage = df["homepage"]
    print(homepage.head())


def convert_json_to_dict(column):
    col = df[column].astype('str')
    col = col.apply(lambda x: ast.literal_eval(x))
    return col


if __name__ == '__main__':
    df = pd.read_csv("train.csv")
    # preprocess_production_countries()
    # preprocess_runtime()
    preprocess_spoken_languages()
    # print(convert_json_to_dict(df.spoken_languages))