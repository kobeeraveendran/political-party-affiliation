from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

df_con = pd.read_table("datasets/conservative_orig.txt", sep = "\n", header = None, error_bad_lines = False)
df_con.columns = ["text"]
df_con.insert(1, "label", [1 for _ in range(df_con.shape[0])], True)

df_lib = pd.read_table("datasets/democrats_orig.txt", sep = "\n", header = None, error_bad_lines = False)
df_lib.columns = ["text"]
df_lib.insert(1, "label", [0 for _ in range(df_lib.shape[0])], True)

df = pd.concat([df_con, df_lib], sort = False)

x = df['text'].values
y = df['label'].values

x_train, x_test, y_train, y_test = \
 train_test_split(x, y, test_size=0.1, random_state=123)

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(x)
xtrain= tokenizer.texts_to_sequences(x_train)
xtest= tokenizer.texts_to_sequences(x_test)

vocab_size=len(tokenizer.word_index)+1

maxlen=10
xtrain=pad_sequences(xtrain,padding='post', maxlen=maxlen)
xtest=pad_sequences(xtest,padding='post', maxlen=maxlen) 
 
print(x_train[:2])
print(x_test[:2])

print(xtrain[:2])
print(xtest[:2])
 
embedding_dim=50
model=Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
         output_dim=embedding_dim,
         input_length=maxlen))
model.add(layers.LSTM(units = 50, return_sequences=True))
model.add(layers.LSTM(units=10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", 
     metrics=['accuracy'])
model.summary()
model.fit(xtrain,y_train, epochs=20, batch_size=32, verbose=True)

loss, acc = model.evaluate(xtrain, y_train, verbose=True)
print("Training Accuracy: {:.2f}".format(acc * 100))
train_preds = model.predict_classes(xtrain)
print("Training F1 score: {:.2f}".format(f1_score(y_train, train_preds)))

loss, acc = model.evaluate(xtest, y_test, verbose=True)
print("Test Accuracy: {:.2f}".format(acc * 100))
test_preds = model.predict_classes(xtest)
print("Test F1 score: {:.2f}".format(f1_score(y_test, test_preds)))

#ypred=model.predict(xtest)

#ypred[ypred>0.5]=1 
#ypred[ypred<=0.5]=0

# result=zip(x_test, y_test, ypred)
# for i in result:
#     print(i)