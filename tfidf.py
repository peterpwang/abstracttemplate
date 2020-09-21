import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import csv
import argparse
import sys
import os

# public variables
debug = 0;


def read_text(text_path):
    data = []
    with open(text_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for row in csv_reader:
            data.append(row[0])
    return data


def create_tfidf(corpus):
    # Calculate TFIDF
    vectorizer = TfidfVectorizer(stop_words='english', 
                                 #min_df=5, max_df=.5, 
                                 ngram_range=(1,1))
    tfidf = vectorizer.fit_transform(corpus)
    #print(tfidf)

    # Get features and index
    features = vectorizer.get_feature_names()
    indices = np.argsort(vectorizer.idf_)[::-1]

    # Print TFIDF values in the first document
    doc = 0
    while doc < 5:
        feature_index = tfidf[doc,:].nonzero()[1]
        tfidf_scores = dict(zip([features[x] for x in feature_index], [tfidf[doc, x] for x in feature_index]))
        text = corpus[doc]

        for w in text.split():
            score = 0.0
            if w in tfidf_scores and tfidf_scores[w] > 0.05:
                print('$' + w + '$', end=' ')
            else:
                print(w, end=' ')

        print('\n')
        doc = doc + 1

    # Get the top 20 features
    top_n = 20
    top_features = [features[i] for i in indices[:top_n]]

    return top_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=0, type=int, metavar='N',
                        help='Debug?') 
    parser.add_argument('--input', default='./data/text/data.csv', metavar='N',
                        help='CVS data file path') 
    parser.add_argument('--output', default='./data/text/tfidf.csv', metavar='N',
                        help='TFIDF data file path') 
    args = parser.parse_args()

    debug = args.debug

    corpus = read_text(args.input)
    features = create_tfidf(corpus)
    #print(features)
