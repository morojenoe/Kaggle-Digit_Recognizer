from pandas import read_csv
from sklearn.ensemble import forest
from sklearn.cross_validation import train_test_split


def apply_random_forest():
    train_data = read_csv("train.csv")
    train_labels = train_data.get("label")
    train_data = train_data.drop("label", axis=1)

    random_forest = forest.RandomForestClassifier(n_estimators=150,
                                                  criterion="gini",
                                                  max_depth=100,
                                                  min_samples_split=2,
                                                  min_samples_leaf=1,
                                                  max_features="auto",
                                                  max_leaf_nodes=None,
                                                  bootstrap=True,
                                                  oob_score=False,
                                                  n_jobs=-1,
                                                  random_state=None,
                                                  verbose=0,
                                                  min_density=None,
                                                  compute_importances=None
                                                  )
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.3)
    random_forest.fit(train_data, train_labels)
    print(random_forest.score(test_data, test_labels))
