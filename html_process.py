#from bs4 import BeautifulSoup
#import stanza

import argparse
import sys
import os
import random

# public variables
number_htmls = 0
debug = 0;


def read_original_data(original_data_dir, text_data_path):

    global number_htmls, debug

    lines = []

    for subdir, dirs, files in os.walk(original_data_dir):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".html"):
                text = convert_html_to_text(filepath)
                lines.append(text)

                number_htmls = number_htmls + 1
                if debug == 1 and number_htmls%1000 == 0:
                    print(".", end = '', flush=True)

    random.shuffle(lines)

    number_train = int(number_htmls * 0.7)
    number_validation = int(number_htmls * 0.2)
    number_test = number_htmls - number_train - number_validation

    f = open(text_data_path + "/train.txt", 'w')
    for i in range(0, number_train):
        f.write(lines[i] + "\n");
    f.close()

    f = open(text_data_path + "/validation.txt", 'w')
    for i in range(number_train, number_train + number_validation):
        f.write(lines[i] + "\n");
    f.close()

    f = open(text_data_path + "/test.txt", 'w')
    for i in range(number_train + number_validation, number_htmls):
        f.write(lines[i] + "\n");
    f.close()

    print(str(number_htmls) +  " (" + str(number_train) + "/" + str(number_validation) + "/" + str(number_test) + ") converted.", flush=True)


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
    parser.add_argument('--input', default='./data/origin/', metavar='N',
                        help='Root path of HTML data files') 
    parser.add_argument('--output', default='./data/text/', metavar='N',
                        help='Output path of text data') 
    args = parser.parse_args()

    debug = args.debug

    read_original_data(args.input, args.output)


#stanza.download('en')       # This downloads the English models for the neural pipeline
#nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English
#doc = nlp(text)
#doc.sentences[0].print_tokens()
#doc.sentences[0].print_words()

