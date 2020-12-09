import numpy as np
from sklearn.utils import shuffle

def load_data():

    data = []

    with open("datasets/conservative_orig.txt", 'r') as file:
        lines = file.readlines()

        for line in lines:
            data.append([line[:-1], 1])

    with open("datasets/democrats_orig.txt", 'r') as file:
        lines = file.readlines()

        for line in lines:
            data.append([line[:-1], 0])

    shuffled = shuffle(data)

    # for example in shuffled[:5]:
    #     print(example[0])
    #     print(example[1])
    #     print()

    cutoff = int(len(shuffled) * 0.8)

    train = shuffled[:cutoff]
    test = shuffled[cutoff:]

    return train, test

if __name__ == "__main__":

    train, test = load_data()

    print(train[:5])
    print('\n')
    print(test[:5])