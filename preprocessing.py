import pandas as pd
import numpy as np
import ast
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk


class Preprocessing:
    mean_val = 0
    genres_ids = set()
    original_languages = []
    top_words_rev = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    top_words_in_overview_mean_rev = dict()
    top_words_votes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    top_words_in_overview_mean_votes = dict()


    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def get_df(self):
        return self.df

    def replace_na_in_tagline(self):
        self.df["tagline"] = self.df["tagline"].fillna("")

    def drop_not_released(self):
        self.df = self.df[self.df.status == "Released"]

    def drop_not_relevant_columns(self):
        not_relavant_columns = ["id",
                                "original_title",
                                "keywords",
                                "title",
                                "tagline",
                                "status"]
        self.df = self.df.drop(columns=not_relavant_columns)

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
        # df = self.get_df()
        self.df['today'] = pd.to_datetime(pd.to_datetime('today').strftime("%Y-%m-%d"))
        self.df['release_date'] = pd.to_datetime(self.df['release_date'])
        self.df['days_till_release'] = (self.df['release_date'] - self.df['today']).dt.days
        self.df = self.df[self.df['days_till_release'] < 0]
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
        self.turn_nan_value_to_mean(['year', 'day', 'month'])
        self.df = self.df.drop('release_date', axis=1)

    def preprocess_budget(self):
        added_cols = self.turn_value_to_nan({'budget': 10}, "<")
        self.turn_nan_value_to_mean(['budget'])

    def preprocess_belongs_to_collection(self):
        self.df['belongs_to_collection'] = self.df['belongs_to_collection'].fillna("")
        self.df['binary_collection'] = (self.df["belongs_to_collection"] != "").astype(int)
        self.df = self.df.drop('belongs_to_collection', axis=1)

    def preprocess_genres(self, train=True):
        self.df['genres'] = self.df.genres.fillna("[]")
        self.convert_json_to_dict("genres")

        if train:
            dics = [dic for dic in self.df["genres"]]
            for lst in dics:
                for dic in lst:
                    Preprocessing.genres_ids.add(dic["name"])

        for key in Preprocessing.genres_ids:
            self.df["genres" + f"_{key}"] = [1 if f"{key}" in string else 0 for string in self.df['genres'].astype('str')]

        self.df.drop(columns=["genres"], inplace=True)

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

    def replace_na_in_overview(self):
        self.df["overview"] = self.df["overview"].fillna("")

    def original_language_feature(self, train=True):
        """
        One-Hot feature original_language,leave columns of language only for language with higher incomes then the average
        :param :
        :return:
        """
        languages = pd.unique(self.df["original_language"])
        revenue_means = [self.df.loc[self.df['original_language'] == lan]["revenue"].mean() for lan in languages]
        if train:
            Preprocessing.mean_val = sum(revenue_means) / len(revenue_means)
            self.df = pd.get_dummies(data=self.df, columns=(["original_language"]))
            for i, one_lan in enumerate(languages):
                if revenue_means[i] >= Preprocessing.mean_val:
                    Preprocessing.original_languages.append(f"original_language_{one_lan}")
                else:
                    self.df.drop(columns=[f"original_language_{one_lan}"], inplace=True)
        else:
            for lan in Preprocessing.original_languages:
                self.df[f"original_language_{lan}"] = (lan == self.df["original_language"]).astype(int)
            self.df.drop(columns=["original_language"], inplace=True)

    def preprocess_overview(self, train=True, revenue=True):
        """
        Extracting all the words, tokenizing them and giving them weights according to revenue and vote_average,
        to know which words affect the most
        """
        if revenue:
            feature_to_process = 'revenue'
        else:
            feature_to_process = 'vote_average'
        stop_words = set(stopwords.words("english"))

        if train:
            nltk.download('punkt')
            s = ""
            for x in self.df["overview"]:
                s += x + "\nto\n" if type(x) == str else ""
            nltk.download("stopwords")
            words_in_s = word_tokenize(s)
            words_dict_4_ = {word: 0 for word in words_in_s if word.casefold() not in stop_words}

            for x, f_t_p in zip(self.df["overview"], self.df[feature_to_process]):
                if type(x) == str and len(x) > 0:
                    tokenized_x = word_tokenize(x + "\nto")
                    tokenized_x = [w for w in tokenized_x if w.casefold() not in stop_words]
                    word_num = len(tokenized_x)
                    for w in tokenized_x:
                        words_dict_4_[w] += (f_t_p* x.count(w)) / word_num
            words_dict_4_ = dict(sorted(words_dict_4_.items(), key=lambda item: item[1]))

            keys_list = list(words_dict_4_.keys())
            tmp = []
            for x in keys_list:
                if len(x) > 2:
                    tmp.append(x)
            keys_list = tmp
            #####    TO SAVE for feature_to_process   ####
            if revenue:
                Preprocessing.top_words_rev[1] = keys_list[-2:]
                Preprocessing.top_words_rev[2] = keys_list[-7:-2]
                Preprocessing.top_words_rev[3] = keys_list[-20:-7]
                Preprocessing.top_words_rev[4] = keys_list[-50:-20]
                Preprocessing.top_words_rev[5] = keys_list[-130:-50]
                Preprocessing.top_words_rev[6] = keys_list[-500:-130]
            else:
                Preprocessing.top_words_votes[1] = keys_list[-2:]
                Preprocessing.top_words_votes[2] = keys_list[-7:-2]
                Preprocessing.top_words_votes[3] = keys_list[-20:-7]
                Preprocessing.top_words_votes[4] = keys_list[-50:-20]
                Preprocessing.top_words_votes[5] = keys_list[-130:-50]
                Preprocessing.top_words_votes[6] = keys_list[-500:-130]


        # adding_to_df
        top_words_in_overview = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        for x in self.df["overview"]:
            if type(x) == str and len(x) > 0:
                tokenized_x = word_tokenize(x + "\nto")
                tokenized_x = [w for w in tokenized_x if w.casefold() not in stop_words]
                word_num = len(tokenized_x)
                for i in range(1, 7):
                    score_i = 0
                    if revenue:
                        tmp_arr = Preprocessing.top_words_rev[i]
                    else:
                        tmp_arr = Preprocessing.top_words_votes[i]
                    for s in tmp_arr:
                        if s in tokenized_x:
                            score_i += x.count(s)
                    score_i *= 1 / word_num
                    top_words_in_overview[i].append(score_i)
            else:
                for i in range(1, 7):
                    top_words_in_overview[i].append(np.nan)

        if train:
            # add to class Attributes feature_to_process
            for i in range(1, 7):
                np_arr = np.array(top_words_in_overview[i])
                if revenue:
                    Preprocessing.top_words_in_overview_mean_rev[i] = np.nanmean(np_arr)
                else:
                    Preprocessing.top_words_in_overview_mean_votes[i] = np.nanmean(np_arr)



        for i in range(1, 7):
            self.df[f'top_words_{feature_to_process}_{str(i)}'] = top_words_in_overview[i]
            if revenue:
                self.df[f'top_words_{feature_to_process}_{str(i)}'].fillna(Preprocessing.top_words_in_overview_mean_rev[i],
                                                                       inplace=True)
            else:
                self.df[f'top_words_{feature_to_process}_{str(i)}'].fillna(Preprocessing.top_words_in_overview_mean_votes[i],
                                                                       inplace=True)


        self.df.drop(columns=["overview"], inplace=True)

    def process_revenue(self, train=True):
        if train:
            self.drop_not_released()
            self.delete_not_released_movies()
        self.original_language_feature(train)
        self.preprocess_date()
        self.preprocess_budget()
        self.preprocess_belongs_to_collection()
        self.preprocess_genres(train)
        self.preprocess_production_countries()
        self.preprocess_production_companies()
        self.preprocess_runtime()
        self.preprocess_spoken_languages()
        self.preprocess_cast()
        self.preprocess_crew()
        self.preprocess_homepage()
        self.preprocess_overview(train, True)
        self.drop_not_relevant_columns()
        return self.df

    def process_vote_avergae(self, train=True):
        if train:
            self.drop_not_released()
            self.delete_not_released_movies()
        # self.original_language_feature(train)
        self.preprocess_date()
        # self.df.drop(columns=["release_date"], inplace=True)
        # self.preprocess_budget()
        self.preprocess_belongs_to_collection()
        self.preprocess_genres(train)
        self.preprocess_production_countries()
        self.preprocess_production_companies()
        self.preprocess_runtime()
        self.preprocess_spoken_languages()
        self.preprocess_cast()
        self.preprocess_crew()
        self.preprocess_overview(train, False)
        # self.preprocess_homepage()
        self.df.drop(columns=["original_language", "budget", "homepage"], inplace=True)
        self.drop_not_relevant_columns()
        return self.df


