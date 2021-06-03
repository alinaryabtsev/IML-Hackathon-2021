import pandas as pd
import ast


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
        
    def preprocess_production_countries(self):
        """
        replace production_countries column with a counter of production_countries
        :return: None
        """
        self.df["production_countries"] = self.df.production_countries.fillna("[]")
        self.convert_json_to_dict("production_countries")
        # add column of production_countries count.
        self.df["production_countries_count"] = self.df.production_countries.apply(len)
        self.df.drop(columns=["production_countries"], inplace=True)

    def preprocess_production_companies(self):
        """
        replace production_companies column with a counter of production_companies
        :return: None
        """
        self.df["production_companies"] = self.df.production_companies.fillna("[]")
        self.convert_json_to_dict("production_companies")

        # add column of production_companies count.
        self.df["production_companies_count"] = self.df.production_companies.apply(len)
        self.df.drop(columns=["production_companies"], inplace=True)

    def preprocess_runtime(self):
        """
        fill missing values of runtime with mean values and make sure running time is reasonable
        :return:
        """
        self.df.runtime = self.df.runtime.where(self.df.runtime.between(1, 1440))
        self.df.runtime = self.df.runtime.fillna(self.df.runtime.mean())

    def preprocess_spoken_languages(self):
        """
        replace spoken_languages column with a counter of spoken_languages
        :return: None
        """
        self.df["spoken_languages"] = self.df.spoken_languages.fillna("[]")
        self.convert_json_to_dict("spoken_languages")

        # add column of language count.
        self.df["spoken_language_count"] = self.df.spoken_languages.apply(len)
        self.df.drop(columns=["spoken_languages"], inplace=True)

    def preprocess_cast(self):
        """
        replace cast column with a counter of cast
        :return: None
        """
        self.df["cast"] = self.df.cast.fillna("[]")
        self.convert_json_to_dict("cast")

        # add column of cast count.
        self.df["cast_count"] = self.df.cast.apply(len)
        self.df.drop(columns=["cast"], inplace=True)

    def preprocess_crew(self):
        """
        replace crew column with a counter of crew
        :return: None
        """
        self.df["crew"] = self.df.crew.fillna("[]")
        self.convert_json_to_dict("crew")

        # add column of crew count.
        self.df["crew_count"] = self.df.crew.apply(len)
        self.df.drop(columns=["crew"], inplace=True)

    def preprocess_homepage(self):
        """
         # change into binary column. 1 for having homepage and 0 for not
        :return:
        """
        self.df["homepage"] = self.df["homepage"].fillna("")
        self.df["binary_homepage"] = (self.df["homepage"] != '').astype(int)

        # get rid of homepage column
        self.df.drop(columns=["homepage"], inplace=True)

    def convert_json_to_dict(self, column):
        self.df[column] = self.df[column].apply(lambda s: list(ast.literal_eval(s)))
