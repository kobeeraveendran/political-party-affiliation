import numpy as np
import spacy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import time

def build_dataset():

    maxlen = 0

    feature_map = {}

    nlp = spacy.load("en_core_web_sm")

    samples = []
    labels = []

    index = 0

    with open("../datasets/conservative.txt", 'r') as file:
        lines = file.readlines()

        # for each training example
        for line in lines:
            # get POS, stop word status, etc. for each word
            doc_tokens = nlp(line)

            curr_doc = [0] * len(feature_map)

            for token in doc_tokens:
                if token.pos_ != 'X' and token.text.isalpha() and not token.is_stop:
                    token_lower = token.text.lower()

                    maxlen = max(maxlen, len(token_lower))

                    if token_lower not in feature_map:
                        feature_map[token_lower] = index
                        index += 1
                        curr_doc.append(1)

                    else:
                        curr_doc[feature_map[token_lower]] += 1

            samples.append(curr_doc)
            labels.append(1)

    with open("../datasets/democrats.txt", 'r') as file:
        lines = file.readlines()

        for line in lines:
            # get POS, stop word status, etc. for each word
            doc_tokens = nlp(line)

            curr_doc = [0] * len(feature_map)

            for token in doc_tokens:
                if token.pos_ != 'X' and token.text.isalpha() and not token.is_stop:
                    token_lower = token.text.lower()

                    if token_lower not in feature_map:
                        feature_map[token_lower] = index
                        index += 1
                        curr_doc.append(1)

                    else:
                        curr_doc[feature_map[token_lower]] += 1

            samples.append(curr_doc)
            labels.append(0)

    for doc in samples[:-1]:
        if len(doc) < len(samples[-1]):
            doc.extend([0] * (len(samples[-1]) - len(doc)))

    samples = np.asarray(samples)
    labels = np.asarray(labels)

    # # index-consistent shuffle of samples and labels
    # samples, labels = shuffle(samples, labels)

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = 0.2, shuffle = True, random_state = 42)

    return X_train, y_train, X_test, y_test, maxlen