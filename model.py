from preprocessing import Preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import pickle5


class Model:
    def __init__(self):
        preprocessing_vote = Preprocessing("train.csv")
        preprocessing_revenue = Preprocessing("train.csv")
        self.vote_matrix = preprocessing_vote.process_vote_avergae()
        self.revenue_matrix = preprocessing_revenue.process_revenue()
        self.revenue_response = self.revenue_matrix["revenue"].to_numpy()
        self.vote_average_response = self.vote_matrix["vote_average"].to_numpy()
        self.vote_matrix.drop(columns=["vote_average"], inplace=True)
        self.revenue_matrix.drop(columns=["revenue"], inplace=True)
        self.vote_regression = None
        self.revenue_regression = None
        self.vote_matrix = self.vote_matrix.to_numpy()
        self.revenue_matrix = self.revenue_matrix.to_numpy()

    def process_test_data(self, csv_file):
        return self.process_test_revenue_data(csv_file), \
               self.process_test_vote_average_data(csv_file)

    def process_test_vote_average_data(self, csv_file):
        preprocessing = Preprocessing(csv_file)
        return preprocessing.process_vote_avergae(False)

    def process_test_revenue_data(self, csv_file):
        preprocessing = Preprocessing(csv_file)
        return preprocessing.process_revenue(False)

    def train_model(self):
        self.vote_regression = RandomForestRegressor(n_estimators=100,
                                                     max_depth=10,
                                                     random_state=0,
                                                     min_samples_split=5).fit(self.vote_matrix,
                                                                              self.vote_average_response)

        self.revenue_regression = RandomForestRegressor(n_estimators=100,
                                                        max_depth=10,
                                                        random_state=0,
                                                        min_samples_split=7).fit(self.revenue_matrix,
                                                                                 self.revenue_response)

    def predict_vote_average(self, samples):
        return self.vote_regression.predict(samples)

    def predict_revenue(self, samples):
        return self.revenue_regression.predict(samples)

    def score(self, y_predicted, y_true):
        return math.sqrt(mean_squared_error(y_true, y_predicted))

    @classmethod
    def serialize_model(cls, obj, filename):
        file = open(filename, 'wb')
        pickle5.dump(obj, file)
        file.close()

    @classmethod
    def deserialize_model(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle5.load(f)
            return obj


