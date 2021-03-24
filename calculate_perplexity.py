""" Calculate perplexity of 1-gram, 2-gram and 3-gram.
"""


import sys
import argparse
import nltk
nltk.download('punkt')

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

from util import read_text


def calculate_perplexity(text):

    train, vocab = padded_everygram_pipeline(2, text)
    lm = MLE(2)
    lm.fit(train, vocab)

    perplexity = (0, lm.perplexity(text), 0)
    return perplexity

    
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
    print(lines)
    perplexity = calculate_perplexity(lines)

    print("Perplexity: ngram-1="+str(perplexity[0])+",  ngram-2="+str(perplexity[1])+",  ngram-3="+str(perplexity[2]))

if __name__ == "__main__":
    main()

