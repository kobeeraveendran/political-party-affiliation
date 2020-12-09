from sklearn.svm import LinearSVC
import numpy as np
import sys

from sklearn.metrics import f1_score

from preprocessing_nb import build_dataset

clf = LinearSVC()

X_train, y_train, X_test, y_test, _, _ = build_dataset()

# print("SHAPES:")
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

print("Dataset loaded")

print("Num training examples: ", X_train.shape[0])
print("Num test examples: ", X_test.shape[0])
print("Num unique words in feature map: ", X_train.shape[1])

clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = clf.score(X_test, y_test)
f1 = f1_score(y_test, preds)

print("Test accuracy: {:.2f}".format(acc * 100))
print("F1 Score: {:.4f}".format(f1))