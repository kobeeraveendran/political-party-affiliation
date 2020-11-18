import numpy as np

def build_dataset():

    text = []
    labels = []
    maxlen = 0

    with open("../datasets/democrats.txt", 'r') as file:
        lines = file.readlines()

        for line in lines:
            maxlen = max(maxlen, len(line.split()))
            text.append(line)
        labels.extend([0 for line in lines])

    with open("../datasets/conservative.txt", 'r') as file:
        lines = file.readlines()

        for line in lines:
            maxlen = max(maxlen, len(line.split()))
            text.append(line)
        labels.extend([1 for line in lines])

    print(text[:3])
    print(labels[:3])

    return text, np.asarray(labels), maxlen