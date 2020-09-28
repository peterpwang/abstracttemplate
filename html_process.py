#from bs4 import BeautifulSoup
#import stanza

import argparse
import sys
import os
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# public variables
debug = 0;


def read_original_data(html_data_dir, text_data_path, tfidf_data_path):

    global debug

    number_htmls = 0
    lines = []

    # Read text from html files and save into a list.
    for subdir, dirs, files in os.walk(html_data_dir):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".html"):
                text = convert_html_to_text(filepath)
                lines.append(text)

                number_htmls = number_htmls + 1
                if debug == 1 and number_htmls%1000 == 0:
                    print(".", end = '', flush=True)

    # Write text file
    f = open(text_data_path + "/data.txt", 'w')
    for line in lines:
        f.write(line)
        f.write("\n");
    f.close()

    # Split and save text into text files
    text_split(lines, text_data_path)

    # Create TFIDF text
    # Split and save text into text files
    lines = create_tfidf(lines, tfidf_data_path)
    text_split(lines, tfidf_data_path)

    print(str(number_htmls) +  " converted.", flush=True)


def text_split(lines, text_path):
    # Split and save text into text files
    number_lines = len(lines)

    random.shuffle(lines)

    number_train = int(number_lines * 0.7)
    number_validation = int(number_lines * 0.2)
    number_test = number_lines - number_train - number_validation

    f = open(text_path + "/train.txt", 'w')
    for i in range(0, number_train):
        f.write(lines[i] + "\n");
    f.close()

    f = open(text_path + "/validation.txt", 'w')
    for i in range(number_train, number_train + number_validation):
        f.write(lines[i] + "\n");
    f.close()

    f = open(text_path + "/test.txt", 'w')
    for i in range(number_train + number_validation, number_lines):
        f.write(lines[i] + "\n");
    f.close()


def create_tfidf(lines, tfidf_text_path):
    # Calculate TFIDF
    vectorizer = TfidfVectorizer(stop_words='english', 
                                 #min_df=5, max_df=.5, 
                                 ngram_range=(1,1))
    tfidf = vectorizer.fit_transform(lines)
    #print(tfidf)

    # Get features and index
    features = vectorizer.get_feature_names()
    indices = np.argsort(vectorizer.idf_)[::-1]

    lines_tfidf = []
    f = open(tfidf_text_path + "/data_tfidf.txt", 'w')

    # Replace words with TFIDF < 0.05 with RRRR
    doc = 0
    for line in lines:
        feature_index = tfidf[doc,:].nonzero()[1]
        tfidf_scores = dict(zip([features[x] for x in feature_index], [tfidf[doc, x] for x in feature_index]))

        text = lines[doc]
        line = ""

        for w in text.split():
            score = 0.0
            if w in tfidf_scores and tfidf_scores[w] > 0.05:
                f.write("RRRR ");
                line = line + "RRRR "
            else:
                f.write(w + " ");
                line = line + w + " "

        f.write("\n");
        lines_tfidf.append(line)

        doc = doc + 1

    f.close()

    # Get the top 20 features
    #top_n = 20
    #top_features = [features[i] for i in indices[:top_n]]

    return lines_tfidf


def convert_html_to_text(html_path):

    f = open(html_path, 'r')
    html = f.read()
    f.close()

    # get text. BS does not work well.
    #soup = BeautifulSoup(html, features="html.parser")
    #text = soup.p.getText()
    idx1 = html.index("<p>") + 3
    idx2 = html.index("</p>")
    return html[idx1:idx2]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=0, type=int, metavar='N',
                        help='Debug?') 
    parser.add_argument('--input_path', default='./data/0/', metavar='N',
                        help='Root path of HTML data files') 
    parser.add_argument('--text_path', default='./data/1/', metavar='N',
                        help='Output path of text data') 
    parser.add_argument('--tfidf_path', default='./data/2/', metavar='N',
                        help='Output path of TFIDF text data') 
    args = parser.parse_args()

    debug = args.debug

    read_original_data(args.input_path, args.text_path, args.tfidf_path)


#stanza.download('en')       # This downloads the English models for the neural pipeline
#nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English
#doc = nlp(text)
#doc.sentences[0].print_tokens()
#doc.sentences[0].print_words()

