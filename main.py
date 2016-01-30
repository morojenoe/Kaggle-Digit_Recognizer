from pandas import read_csv
from prepare_data import *
from draw_digit import show_digit_from_df
import time
import pickle


if __name__ == "__main__":
    time.clock()
    # train_data = read_csv("train.csv")
    # pickle.dump(train_data, open("train_data.pkl", 'wb'))
    train_data = pickle.load(open("train_data.pkl", 'rb'))
    # train_data = reduce_df_to_01(train_data)
    # pickle.dump(train_data, open("train_data.pkl", 'wb'))
    pattern = get_patterns(train_data)

    train_data = apply_pattern(train_data, pattern)

    for i in range(0, 10):
        show_digit_from_df(train_data, i)
