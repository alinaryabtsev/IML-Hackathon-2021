
################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
from model import Model
from preprocessing import Preprocessing


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    arr = Model.deserialize_model("train_serialized.pkl")
    model = arr[0]
    Preprocessing.mean_val = arr[1]
    Preprocessing.genres_ids = arr[2]
    Preprocessing.original_languages = arr[3]
    Preprocessing.top_words_rev = arr[4]
    Preprocessing.top_words_in_overview_mean_rev = arr[5]
    Preprocessing.top_words_votes = arr[6]
    Preprocessing.top_words_in_overview_mean_votes = arr[7]

    # model = Model()
    # model.train_model()

    revenue_data, vote_average_data = model.process_test_data(csv_file)
    revenue_true = revenue_data["revenue"]
    vote_true = vote_average_data["vote_average"]
    vote_average_data.drop(columns=["vote_average"], inplace=True)
    revenue_data.drop(columns=["revenue"], inplace=True)
    y_hat_revenue = model.predict_revenue(revenue_data.to_numpy())
    print(model.score(y_hat_revenue, revenue_true.to_numpy()))
    y_hat_votes = model.predict_vote_average(vote_average_data.to_numpy())
    print(model.score(y_hat_votes, vote_true.to_numpy()))
    return (y_hat_revenue, y_hat_votes)


if __name__ == '__main__':
    predict("test.csv")
