import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.8
TEST_RATIO = 0.2


def combine_files(first, sec):
    df_1 = pd.read_csv(first)
    df_2 = pd.read_csv(sec)
    return pd.concat([df_1, df_2], axis=0, ignore_index=True)


def split_data():
    df = combine_files("movies_dataset.csv", "movies_dataset_part2.csv")

    col1 = "vote_average"
    col2 = "revenue"

    y = df[[col1, col2]]
    X = df.drop([col1, col2], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=5)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)




if __name__ == '__main__':
    split_data()



    










