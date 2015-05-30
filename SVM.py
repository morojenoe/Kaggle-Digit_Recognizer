from sklearn import svm
from sklearn.cross_validation import train_test_split


def apply_svm(train_data, train_labels):
    classificator = svm.SVC()
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.3)
    classificator.fit(train_data, train_labels)
    print(classificator.score(test_data, test_labels))
