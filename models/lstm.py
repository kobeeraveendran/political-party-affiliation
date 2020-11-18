from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../")

from preprocessing_lstm import build_dataset

embedding_dim = 50

model = Sequential()

# X_train, y_train, X_test, y_test, maxlen, text = build_dataset()

text, labels, maxlen = build_dataset()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size = 0.2, random_state = 42)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, padding = 'post', maxlen = maxlen)
x_test = pad_sequences(x_test, padding = 'post', maxlen = maxlen)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

vocab_size = x_train.shape[1]

model.add(layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = maxlen))
model.add(layers.LSTM(units = 50, return_sequences = True))
model.add(layers.LSTM(units = 10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8))
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.summary()

model.fit(x_train, y_train, epochs = 20, batch_size = 20)
loss, acc = model.evaluate(x_train, y_train, verbose = False)
print("Training acc: ", acc.round(2))
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = True)
print("Test acc: ", test_acc.round(2))

# y_preds = model.predict(X_test)
