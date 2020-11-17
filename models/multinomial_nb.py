from sklearn.naive_bayes import MultinomialNB
import numpy as np
import sys

sys.path.append('../')

from preprocessing_nb import build_dataset

clf = MultinomialNB(alpha = 1.0, fit_prior = True)

X_train, y_train, X_test, y_test = build_dataset()

print("SHAPES:")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print("Dataset loaded")

print("Num training examples: ", X_train.shape[0])
print("Num test examples: ", X_test.shape[0])
print("Num unique words in feature map: ", X_train.shape[1])

clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = clf.score(X_test, y_test)

print("Test accuracy: {:.2f}".format(acc * 100))