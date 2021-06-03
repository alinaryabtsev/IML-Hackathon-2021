"""
Analyse features: original_language, original_title, overview, vote_count, production_companies
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from preprocessing import Preprocessing
import json as js
import ast
import numpy as np
import sys
from pandas.core.frame import DataFrame


def order_titles_by_length(preprocessing):
    df = preprocessing.get_df()
    lens = df.title.str.len().sort_values().index
    titles = df["title"].reindex(lens)
    return titles


def plot_revenue_or_vote_average_over_length_of_text(df, feature, what):
    """
    plots revenue or vote average over length of the feature
    :param df: movies data feame
    :param feature: featue
    :param what: revenue or vote average
    """
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    df["len"] = df[feature].str.len()
    all_lens = df[feature].str.len().unique()
    means = df.groupby("len")[what].mean()
    counts = df.groupby("len")[what].count().reset_index()
    chart = sns.barplot(x=all_lens, y=means, data=counts)
    plt.xlabel(f'{feature} lengths')
    for i, p in enumerate(chart.patches):
        chart.annotate("%.0f" % counts[what][i],
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                       textcoords='offset points')
    plt.savefig(f"{what}_over_{feature}_length.png")
    plt.clf()


def plot_corrolation(preprocessing):
    df = preprocessing.get_df()
    correlations = df.corr()
    sns.heatmap(correlations)
    plt.show()


def length_of_title_effect(preprocessing):
    df = preprocessing.get_df()
    plot_revenue_or_vote_average_over_length_of_text(df, "title", "revenue")
    plot_revenue_or_vote_average_over_length_of_text(df, "title", "vote_average")


def length_of_tagline_effect(preprocessing):
    preprocessing.replace_na_in_tagline()
    df = preprocessing.get_df()
    plot_revenue_or_vote_average_over_length_of_text(df, "tagline", "revenue")
    plot_revenue_or_vote_average_over_length_of_text(df, "tagline", "vote_average")

def preprocess_production_countries(preprocessing):
    df = preprocessing.get_df()
    countries = df["production_countries"]
    print(countries.head())


def preprocess_runtime(preprocessing):
    df = preprocessing.get_df()
    df.runtime = df.runtime.where(df.runtime.between(1, 1440))
    sns.barplot(df.runtime, df.revenue)
    # plt.bar(df.runtime, df.revenue)
    # creating the bar plot
    plt.show()


def preprocess_spoken_languages():
    # print(df.spoken_languages.head())
    pass


def preprocess_homepage(preprocessing):
    df = preprocessing.get_df()
    homepage = df["homepage"]
    print(homepage.head())


def convert_json_to_dict(preprocessing, column):
    df = preprocessing.get_df()
    col = df[column].astype('str')
    col = col.apply(lambda x: ast.literal_eval(x))
    return col
  

def original_language_plot(preprocessing, response):
    """
    Plots mean of response={revenue,vote_average} as a function of original_language
    :param preprocessing:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    languages = pd.unique(preprocessing.df["original_language"])
    response_means = [preprocessing.df.loc[preprocessing.df['original_language'] == lan][response].mean() for lan in
                      languages]
    plt.bar(languages, response_means)
    mean_val = sum(response_means) / len(response_means)
    ax.axhline(mean_val)
    plt.savefig(r"C:\Users\morga\IML-Hackathon-2021\plot_revenue_language.png")
    plt.clf()


def vote_count_plot(preprocessing):
    """
    Plots revenue as a function of vote_count
    :param df_train:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(preprocessing.df[:]["vote_count"], preprocessing.df[:]["revenue"])
    ax.set_xlabel('vote_count')
    ax.set_ylabel('revenue')
    ax.legend()
    plt.savefig(r"C:\Users\morga\IML-Hackathon-2021\plot_vote_count.png")
    plt.clf()
  
  
def main():
    preprocessing = Preprocessing("train.csv")

    # length_of_tagline_effect(preprocessing)
    # length_of_title_effect(preprocessing)
    # plot_corrolation(preprocessing)
    # length_of_tagline_effect(preprocessing)
    # plot_corrolation(preprocessing)
    
    response = "vote_average"
    original_language_plot(preprocessing, response)
    vote_count_plot(preprocessing)

if __name__ == '__main__':
    main()
