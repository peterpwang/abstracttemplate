import csv
import argparse
import sys
import os
import stanza

from util import read_text


def create_upos(corpus, output_path):

    #stanza.download('en')
    nlp = stanza.Pipeline('en')

    # Write HTML file
    f = open(output_path, 'w')
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
    while i < 5:
        doc = nlp(corpus[i])
        for sentence in doc.sentences:
            for token in sentence.tokens:
                if token.ner != "O":
                    f.write('<span class="entity">' + token.text + '</span> ')
                else:
                    f.write(token.text + ' ')
            f.write("<br>\n");
        #print(upos.entities)
        print('\n')
        i = i + 1

    f.write("  </body>\n")
    f.write("</html>\n")
    f.write("\n");
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/1/data.txt', metavar='N',
                        help='Text data file path') 
    parser.add_argument('--output', default='./data/2/data_upos.html', metavar='N',
                        help='UPOS tagged HTML file path') 
    args = parser.parse_args()

    corpus = read_text(args.input)
    create_upos(corpus, args.output)
    #print(features)
