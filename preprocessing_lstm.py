import numpy as np
import spacy


def build_dataset():

    text = []
    labels = []
    maxlen = 0

    nlp = spacy.load("en_core_web_sm")

    with open("../datasets/democrats.txt", 'r') as file:
        lines = file.readlines()

        curr_line = []

        for line in lines:
            
            doc_tokens = nlp(line)

            for token in doc_tokens:
                if token.pos_ != 'X' and token.text.isalpha() and not token.is_stop:
                    token_lower = token.text.lower()

                    

            maxlen = max(maxlen, line)
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