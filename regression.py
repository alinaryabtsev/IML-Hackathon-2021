
################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
from model import Model


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    model = Model()
    model.train_model()
    df = model.process_test_data(csv_file)
    revenue_true = df["revenue"]
    vote_true = df["vote_average"]
    df.drop(columns=["revenue", "vote_average"], inplace=True)
    print(model.predict_revenue(df.to_numpy(), revenue_true.to_numpy()))
    print(model.predict_vote_average(df.to_numpy(), vote_true.to_numpy()))


if __name__ == '__main__':
    predict("validation.csv")
