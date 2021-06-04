"""
Serialization of the linear regression
"""

from model import Model
from preprocessing import Preprocessing


model = Model()
model.train_model()

to_ser = [model, Preprocessing.mean_val, Preprocessing.genres_ids, Preprocessing.original_languages, Preprocessing.top_words_rev, Preprocessing.top_words_in_overview_mean_rev, Preprocessing.top_words_votes, Preprocessing.top_words_in_overview_mean_votes]
Model.serialize_model(to_ser, "train_serialized.pkl")


