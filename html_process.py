#from bs4 import BeautifulSoup
#import stanza

import argparse
import sys
import os

# public variables
number_htmls = 0
debug = 0;


def read_original_data(original_data_dir, cvs_data_path):

    global number_htmls, debug

    f = open(cvs_data_path, 'w')

    for subdir, dirs, files in os.walk(original_data_dir):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".html"):
                text = convert_html_to_text(filepath)
                f.write(text);
                f.write("\n");

                number_htmls =number_htmls + 1
                if debug == 1 and number_htmls%1000 == 0:
                    print(".", end = '', flush=True)
    f.close()

    print(str(number_htmls) +  " converted.", flush=True)


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
    parser.add_argument('--output', default='./data/text/data.cvs', metavar='N',
                        help='Output CVS data file path') 
    args = parser.parse_args()

    debug = args.debug

    read_original_data(args.input, args.output)


#stanza.download('en')       # This downloads the English models for the neural pipeline
#nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English
#doc = nlp(text)
#doc.sentences[0].print_tokens()
#doc.sentences[0].print_words()

