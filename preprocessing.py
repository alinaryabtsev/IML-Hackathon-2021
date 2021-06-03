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

    def drop_not_released(self):
        self.df = self.df[self.df.status == "Released"]

    def drop_not_relevant_columns(self):
        not_relavant_columns = ["original title",
                                "overview",
                                "keywords",
                                "title",
                                "tagline",
                                "status"]
        self.df = self.df.drop(columns=not_relavant_columns)

