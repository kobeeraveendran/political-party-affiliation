from sklearn.naive_bayes import MultinomialNB
import numpy as np
import sys

from sklearn.metrics import f1_score

from preprocessing_nb import build_dataset

clf = MultinomialNB(alpha = 1.0, fit_prior = True)

print("Loading dataset...")

X_train, y_train, X_test, y_test, maxlen, feature_map = build_dataset()

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

# to test on an input string...

# test_string = "<test_string_here>"
# test_input = np.zeros(X_train[0].shape)

# for word in test_string.split():
#     if word in feature_map:
#         test_input[feature_map[word]] = 1
# pred = clf.predict([test_input])

# print("Input: {}".format(test_string))
# print("Prediction (0 = lib, 1 = cons): ", pred)