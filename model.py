from preprocessing import Preprocessing
from sklearn.linear_model import LinearRegression


class Model:
    def __init__(self):
        preprocessing = Preprocessing("train.csv")
        self.matrix = preprocessing.get_df()
        self.revenue_response = self.matrix["revenue"].to_numpy()
        self.vote_average_response = self.matrix["vote_average"].to_numpy()
        self.vote_regression = None
        self.revenue_regression = None
        self.matrix = self.matrix.to_numpy()

    def train_model(self):
        self.revenue_regression = LinearRegression().fit(self.matrix, self.revenue_response)
        self.vote_regression = LinearRegression().fit(self.matrix, self.vote_average_response)

    def predict_vote_average(self, samples):
        self.vote_regression.predict(samples)

    def predict_revenue(self, samples):
        self.revenue_regression.predict(samples)
