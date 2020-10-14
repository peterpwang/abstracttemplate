#from bs4 import BeautifulSoup

import argparse
import sys
import os
import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import stanza

from util import read_text

# public variables
debug = 0
skip_extraction = 1
skip_upos = 1
skip_tfidf = 1
skip_first_sentence = 1

sentence_max_word_count = 75
sentence_min_word_count = 6

def read_original_data(html_data_dir, text_data_dir, upos_data_dir, tfidf_data_dir, first_sentence_dir):
    global skip_extraction
    global skip_upos
    global skip_tfidf
    global skip_first_sentense

    if skip_extraction == 0:
        # Read text from html files and save into a list.
        print("Extracting...", flush=True)
        lines = extract_text(html_data_dir, text_data_dir)
        number_htmls = len(lines)
        print(str(number_htmls) +  " extracted.", flush=True)

        # Split and save text into text files
        text_split(lines, text_data_dir)
        print("Datasets created.", flush=True)
    else:
        print("Reading...", flush=True)
        lines = read_text(text_data_dir + "/data.txt")
        number_htmls = len(lines)
        print(str(number_htmls) +  " read from dataset.", flush=True)

    if skip_upos == 0:
        print("Tagging...", flush=True)
        lines = create_upos(lines, upos_data_dir + "/data_upos.txt", upos_data_dir + "/data_upos.html")
        print("UPOS tags created.", flush=True)
    else:
        print("UPOS tags reading...", flush=True)
        lines = read_text(upos_data_dir + "/data_upos.txt")
        print(str(number_htmls) +  " UPOS tagged read from dataset.", flush=True)

    # Create TFIDF text and split and save text into text files
    if skip_tfidf == 0:
        print("TFIDF calculating...", flush=True)
        lines = create_tfidf(lines, tfidf_data_dir)

        # Split and save text into text files
        text_split(lines, tfidf_data_dir)
        print(str(number_htmls) +  " TFIDF calculated.", flush=True)
    else:
        print("TFIDF reading...", flush=True)
        lines = read_text(tfidf_data_dir + "/data_tfidf.txt")
        print(str(number_htmls) +  " TFIDF read from dataset.", flush=True)

    # Create first sentence text and split and save text into text files
    if skip_first_sentence == 0:
        print("Extracting first sentence...", flush=True)
        lines = create_first_sentence(lines, first_sentence_dir)

        # Sort by word counts
        lines = output_sorted_sentence_by_words_count(lines, first_sentence_dir)

        # Split and save text into text files
        text_split(lines, first_sentence_dir)
        print(str(len(lines)) +  " first sentence text extracted.", flush=True)


def extract_text(html_data_dir, text_data_path):
    global debug

    lines = []
    number_htmls = 0

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

    return lines


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


def sort_by_words_count(sentence):
    line_words = sentence.split(" ")
    return len(line_words)


def output_sorted_sentence_by_words_count(lines, text_path):
    global sentence_max_word_count
    global sentence_min_word_count 

    # Sort
    lines_sorted = lines[:]
    lines_sorted.sort(key=sort_by_words_count)

    # Filter out sentencs too short or too long
    lines_new = []
    for i in range(0, len(lines_sorted)):
        line_words = lines_sorted[i].split(" ")
        if (len(line_words) <= sentence_max_word_count and len(line_words) >= sentence_min_word_count):
            lines_new.append(lines_sorted[i])

    # Write to a file
    f = open(text_path + "/data_sorted_by_len.txt", 'w')
    for i in range(0, len(lines_new)):
        f.write(lines_new[i] + "\n");
    f.close()

    return lines_new


def create_upos(lines, upos_text_file, upos_html_file):
    global debug

    #stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

    docs = []
    i = 0
    for line in lines:
        doc = nlp(line)
        docs.append(doc)
        i = i + 1
        if debug == 1 and i%1000 == 0:
            print(".", end = '', flush=True)

    # Output 5 documents into sample HTML file
    f = open(upos_html_file, 'w')
    f.write("<!DOCTYPE html>\n")
    f.write("<html lang=\"en\">\n")
    f.write("  <head>\n")
    f.write("    <meta charset=\"utf-8\">\n")
    f.write("    <style>\n")
    f.write("      .entity { color: green; }\n")
    f.write("    </style>\n")
    f.write("  </head>\n")
    f.write("  <body>\n")

    i = 0
    for doc in docs:
        if i >= 5:
            break

        for sentence in doc.sentences:
            for token in sentence.tokens:
                if token.ner != "O":
                    f.write('<span class="entity">' + token.text + '</span> ')
                else:
                    f.write(token.text + ' ')

        f.write("<br>\n");
        i = i + 1

    f.write("  </body>\n")
    f.write("</html>\n")
    f.write("\n");
    f.close()

    # Export documents into plain text
    f = open(upos_text_file, 'w')
    for doc in docs:
        for sentence in doc.sentences:
            previous_ner = False
            for token in sentence.tokens:
                if token.ner != "O":
                    if not previous_ner:
                        f.write('NNNN ')
                        previous_ner = True
                else:
                    f.write(token.text + ' ')
                    previous_ner = False
        f.write("\n");
    f.close()

    # return in list
    ner_list = []
    for doc in docs:
        s = ""
        for sentence in doc.sentences:
            previous_ner = False
            for token in sentence.tokens:
                if token.ner != "O":
                    if not previous_ner:
                        s += 'NNNN '
                        previous_ner = True
                else:
                    s += token.text + ' '
                    previous_ner = False
        ner_list.append(s)

    return ner_list


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

        previous_tfidf = False
        for w in text.split():
            if w in tfidf_scores and tfidf_scores[w] > 0.05:
                if not previous_tfidf:
                    f.write("RRRR ");
                    line = line + "RRRR "
                    previous_tfidf = True
            else:
                f.write(w + " ");
                line = line + w + " "
                previous_tfidf = False

        f.write("\n");
        lines_tfidf.append(line)

        doc = doc + 1

    f.close()

    # Get the top 20 features
    #top_n = 20
    #top_features = [features[i] for i in indices[:top_n]]

    return lines_tfidf


def create_first_sentence(lines, first_sentence_text_path):
    global debug

    # parse the text
    nlp = stanza.Pipeline(lang='en', processors='tokenize')

    docs = []
    i = 0
    for line in lines:
        doc = nlp(line)
        docs.append(doc)
        i = i + 1
        if debug == 1 and i%1000 == 0:
            print(".", end = '', flush=True)

    # Split the first sentence
    lines_first_sentence = []
    f = open(first_sentence_text_path + "/data_first_sentence.txt", 'w')

    for doc in docs:
        for sentence in doc.sentences:
            lines_first_sentence.append(sentence.text)
            f.write(sentence.text)
            f.write("\n")
            break

    f.close()
    return lines_first_sentence


def convert_html_to_text(html_path):

    f = open(html_path, 'r')
    html = f.read()
    f.close()

    # get text. BS does not work well.
    #soup = BeautifulSoup(html, features="html.parser")
    #text = soup.p.getText()
    idx1 = html.index("<p>") + 3
    idx2 = html.index("</p>")
    s = html[idx1:idx2]

    # Replace format tags in upper and lower cases: sup, bold, italic, super, i, em, sub
    p = re.compile('(</?sup>|</?bold>|</?italic>|</?super>|</?i>|</?em>|</?sub>)')
    s = p.sub('', s)
    return s


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=0, type=int, metavar='N',
                        help='Debug?') 
    parser.add_argument('--input_path', default='./data/0/', metavar='N',
                        help='Root path of HTML data files') 
    parser.add_argument('--text_path', default='./data/1/', metavar='N',
                        help='Output path of text data') 
    parser.add_argument('--upos_path', default='./data/2/', metavar='N',
                        help='Output path of UPOS tagged text data') 
    parser.add_argument('--tfidf_path', default='./data/3/', metavar='N',
                        help='Output path of TFIDF tagged text data') 
    parser.add_argument('--first_sentence_path', default='./data/4/', metavar='N',
                        help='Output path of first sentence text data') 
    parser.add_argument('--skip_extraction', default=1, type=int, metavar='N',
                        help='Skip HTML extraction?') 
    parser.add_argument('--skip_upos', default=1, type=int, metavar='N',
                        help='Skip UPOS creation?') 
    parser.add_argument('--skip_tfidf', default=1, type=int, metavar='N',
                        help='Skip TF/IDF tagging?') 
    parser.add_argument('--skip_first_sentence', default=1, type=int, metavar='N',
                        help='Skip creating first sentence?') 
    args = parser.parse_args()

    debug = args.debug
    skip_extraction = args.skip_extraction
    skip_upos = args.skip_upos
    skip_tfidf = args.skip_tfidf
    skip_first_sentence = args.skip_first_sentence

    read_original_data(args.input_path, args.text_path, args.upos_path, args.tfidf_path, args.first_sentence_path)

