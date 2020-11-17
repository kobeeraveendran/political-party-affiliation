from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers

import sys
sys.path.append("../")

from preprocessing_nb import build_dataset

embedding_dim = 50

model = Sequential()

X_train, y_train, X_test, y_test, maxlen = build_dataset()

vocab_size = X_train.shape[1]

model.add(layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = maxlen))
model.add(layers.LSTM(units = 50, return_sequences = True))
model.add(layers.LSTM(units = 10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8))
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.summary()

model.fit(X_train, y_train, epochs = 20, batch_size = 20)
loss, acc = model.evaluate(X_train, y_train, verbose = False)
print("Training acc: ", acc.round(2))
test_loss, test_acc = model.evaluate(X_test, y_test, verbose = True)
print("Test acc: ", test_acc.round(2))

# y_preds = model.predict(X_test)
