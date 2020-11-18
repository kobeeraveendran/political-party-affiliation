from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split

import numpy as np

import sys
sys.path.append("../")

from preprocessing_lstm import build_dataset

embedding_dim = 50

model = Sequential()

# X_train, y_train, X_test, y_test, maxlen, text = build_dataset()

text, labels, maxlen = build_dataset()

tokenizer = Tokenizer(num_words = 50)
tokenizer.fit_on_texts(text)

text = tokenizer.texts_to_sequences(text)
text = pad_sequences(text, maxlen = maxlen)

x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size = 0.2, random_state = 42)

# x_train = tokenizer.texts_to_sequences(x_train)
# x_test = tokenizer.texts_to_sequences(x_test)

# x_train = pad_sequences(x_train)
# x_test = pad_sequences(x_test)

print("SHAPE X_TRAIN: ", x_train.shape)
print("SHAPE Y_TRAIN: ", y_train.shape)
print("SHAPE X_TEST: ", x_test.shape)

vocab_size = x_train.shape[1]

model.add(layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim))
model.add(layers.LSTM(units = 50, return_sequences = True))
model.add(layers.LSTM(units = 10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8))
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.summary()

model.fit(x_train, y_train, epochs = 20, batch_size = 20)
loss, acc = model.evaluate(x_train, y_train, verbose = False)
print("Training acc: {:.2f}".format(acc * 100))
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = True)
print("Test acc: {:.2f}".format(test_acc * 100))

# y_preds = model.predict(X_test)
