import numpy as np


def reduce_df_to_01(data):
    labels = data.get("label")
    data = data.drop("label", axis=1)
    data = data.applymap((lambda x: 0 if x < 128 else 1))
    data["label"] = labels
    return data


def get_patterns(data):
    grouped = data.groupby(by=("label", ))
    return grouped.aggregate(np.prod)


def apply_pattern_to_row(row, pattern):
    p = pattern.loc[row["label"]]
    p["label"] = 1
    result = row * p
    result["label"] = row["label"]
    return result


def apply_pattern(data, pattern):
    return data.apply(apply_pattern_to_row, axis=1, args=(pattern, ))
