import pandas as pd
import numpy as np
import matplotlib.pyplot as pplt
import preprocessing
from sklearn.feature_extraction.text import CountVectorizer


class Preprocessing:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def get_df(self):
        return self.df

    def replace_na_in_tagline(self):
        self.df["tagline"] = self.df["tagline"].fillna("")

    def replace_na_in_overview(self):
        self.df["overview"] = self.df["overview"].fillna("")

    def original_language_feature(self):
        """
        One-Hot feature original_language,leave columns of language only for language with higher incomes then the average
        :param :
        :return:
        """

        self.df = pd.get_dummies(data=self.df, columns=(["original_language"]))
        languages = pd.unique(self.df["original_language"])
        revenue_means = [self.df.loc[self.df['original_language'] == lan]["revenue"].mean() for lan in languages]
        mean_val = sum(revenue_means) / len(revenue_means)
        for one_lan in languages:
            if one_lan < mean_val:
                self.df.drop([f"original_language_{one_lan}"])
