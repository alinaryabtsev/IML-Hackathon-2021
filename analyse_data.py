"""
Analyse features: original_language, original_title, overview, vote_count, production_companies
"""
import splitting_data
import pandas as pd
import numpy as np
import json
import matplotlib as plt
import matplotlib.pyplot as pplt


def original_language(df_train):
    """
    One-Hot feature original_language
    :param df_train:
    :return:
    """
    df_train = pd.get_dummies(data=df_train, columns=(["original_language"]))
    df_train.to_csv('C:\\Users\\morga\\IML-Hackathon-2021\\train.csv')


def vote_count(df_train):
    """
    Plots revenue as a function of vote_count
    :param df_train:
    :return:
    """
    fig = pplt.figure()
    ax = fig.add_subplot(111)
    pplt.scatter(df_train[:]["vote_count"], df_train[:]["revenue"])
    ax.set_xlabel('vote_count')
    ax.set_ylabel('revenue')
    ax.legend()
    pplt.show()


# def origin_title(df_train):
#     fig = pplt.figure()
#     ax = fig.add_subplot(111)
#     pplt.scatter(df_train[:]["vote_count"], df_train[:]["revenue"])
#     ax.set_xlabel('vote_count')
#     ax.set_ylabel('revenue')
#     ax.legend()
#     pplt.show()


if __name__ == "__main__":
    df_train = pd.read_csv('C:\\Users\\morga\\IML-Hackathon-2021\\train.csv')
    original_language(df_train)
    vote_count(df_train)
