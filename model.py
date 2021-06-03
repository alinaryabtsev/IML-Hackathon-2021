from preprocessing import Preprocessing
from sklearn.ensemble import RandomForestRegressor
import math


class Model:
    def __init__(self):
        preprocessing = Preprocessing("train.csv")
        self.matrix = preprocessing.process_all()
        self.revenue_response = self.matrix["revenue"].to_numpy()
        self.vote_average_response = self.matrix["vote_average"].to_numpy()
        self.matrix.drop(columns=["vote_average", "revenue"], inplace=True)
        self.vote_regression = None
        self.revenue_regression = None
        self.matrix = self.matrix.to_numpy()
        self.vote_hat = None
        self.revenue_hat = None

    def process_test_data(self, csv_file):
        preprocessing = Preprocessing(csv_file)
        return preprocessing.process_all(False)

    def train_model(self):
        self.vote_regression = RandomForestRegressor(max_depth=180, random_state=0).fit(self.matrix,
                                                                                        self.vote_average_response)
        self.revenue_regression = RandomForestRegressor(max_depth=180, random_state=0).fit(self.matrix,
                                                                                           self.revenue_response)

    def predict_vote_average(self, samples, y_true):
        return math.sqrt(self.vote_regression.score(samples, y_true))

    def predict_revenue(self, samples, y_true):
        return math.sqrt(self.revenue_regression.score(samples, y_true))


