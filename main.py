from pandas import read_csv
from prepare_data import *
from draw_digit import show_digit_from_df


if __name__ == "__main__":
    train_data = read_csv("train.csv")
    train_data = reduce_df_to_01(train_data)
    pattern = get_patterns(train_data)

    train_data = apply_pattern(train_data, pattern)

    for i in range(0, 10):
        show_digit_from_df(train_data, i)
