""" Calculate perplexity of 1-gram, 2-gram and 3-gram.
"""


import sys
import argparse
import nltk
nltk.download('punkt')

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

import numpy as np

from util import read_text

def unigram(tokens):    
    model = {}
    for f in tokens:
        if f in model:
            model[f] += 1
        else:
            model[f] = 1
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N
    return model

def calculate_perplexity(text):
    tokens = nltk.word_tokenize(text.lower())
    model = unigram(tokens)

    perplexity = 0
    N = 0

    for word in tokens:
        if word in model:
            N += 1
            perplexity = perplexity + np.log2(model[word])
    perplexity = np.power(-perplexity/N, 2)
    return perplexity

def calculate_perplexity_nltk(lines):

    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
                for sent in lines]

    n = 2
    train, vocab = padded_everygram_pipeline(n, tokenized_text)
    lm = MLE(n)
    lm.fit(train, vocab)

    test, _ = padded_everygram_pipeline(n, tokenized_text)
    perplexities = []
    for i, line in enumerate(test):
        perplexities.append(lm.perplexity(line))
    return perplexities
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="generated_text.txt",
        help="path to generated text",
    )

    args = parser.parse_args()

    # Read text
    lines = read_text(args.input)

    perplexities = calculate_perplexity_nltk(lines)
    for x in perplexities:
        print("PP(NLTK 2-gram):{0}".format(x))

    for text in lines:
        perplexity = calculate_perplexity(text)
        print("PP:{0}".format(perplexity))


if __name__ == "__main__":
    main()

