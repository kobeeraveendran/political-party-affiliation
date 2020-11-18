from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from preprocessing_nb import build_dataset

# clf = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth = 1), 
#     n_estimators = 200
# )

clf = AdaBoostClassifier(
    n_estimators = 200, learning_rate = 1
)

X_train, y_train, X_test, y_test, _ = build_dataset()

clf.fit(X_train, y_train)

preds = clf.predict(X_test)

acc = clf.score(X_test, y_test)

print("AdaBoost")

print("Test accuracy: {:.2f}".format(acc * 100))