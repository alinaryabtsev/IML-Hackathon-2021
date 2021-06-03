import pandas as pd


class Preprocessing:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def get_df(self):
        return self.df

    def replace_na_in_tagline(self):
        self.df["tagline"] = self.df["tagline"].fillna("")

    def replace_na_in_title(self):
        self.df["title"] = self.df["title"].fillna("")


