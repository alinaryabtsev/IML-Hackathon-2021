"""
Analyse features: original_language, original_title, overview, vote_count, production_companies
"""
import preprocessing
import numpy as np
import matplotlib.pyplot as pplt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def original_language_plot(preprocessing, response):
    """
    Plots mean of response={revenue,vote_average} as a function of original_language
    :param preprocessing:
    :return:
    """
    fig, ax = pplt.subplots(figsize=(10, 7))
    languages = pd.unique(preprocessing.df["original_language"])
    response_means = [preprocessing.df.loc[preprocessing.df['original_language'] == lan][response].mean() for lan in
                      languages]
    pplt.bar(languages, response_means)
    mean_val = sum(response_means) / len(response_means)
    ax.axhline(mean_val)
    pplt.savefig(r"C:\Users\morga\IML-Hackathon-2021\plot_revenue_language.png")
    pplt.clf()


def vote_count_plot(preprocessing):
    """
    Plots revenue as a function of vote_count
    :param df_train:
    :return:
    """
    fig = pplt.figure()
    ax = fig.add_subplot(111)
    pplt.scatter(preprocessing.df[:]["vote_count"], preprocessing.df[:]["revenue"])
    ax.set_xlabel('vote_count')
    ax.set_ylabel('revenue')
    ax.legend()
    pplt.savefig(r"C:\Users\morga\IML-Hackathon-2021\plot_vote_count.png")
    pplt.clf()


if __name__ == "__main__":
    preprocessing = preprocessing.Preprocessing(r"C:\Users\morga\IML-Hackathon-2021\train.csv")
    response = "vote_average"
    original_language_plot(preprocessing, response)
    vote_count_plot(preprocessing)
