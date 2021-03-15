import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper libraries
import os
import time
import math
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

import data
import model

# Public variables
device = "cuda"


# Text generator
class Generator():
    
    def __init__(self, args):

        self.data_path = args.data_path
        self.model_file = args.model_file
        self.output_path = args.output_path
        self.num_sentences = args.num_sentences
        self.first_word = args.first_word

        self.corpus = None

        self.seed = 1111
        self.log_interval = 200
        self.words_per_sentence = 10
        self.temperature = 1.0

        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.seed)

    def run(self):

        # GPU related settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch:", torch.__version__, "(CPU+GPU)" if  torch.cuda.is_available() else "(CPU)")
        torch.cuda.set_device(0)

        with open(self.model_file, 'rb') as f:
            model = torch.load(f).to(device)
        model.eval()

        print("Model read.")

        corpus = data.Corpus(self.data_path)
        ntokens = len(corpus.dictionary)

        with open(self.output_path + "/generated.txt", 'w') as outf:
            with torch.no_grad():  # no tracking history
                for j in range(self.num_sentences):
                    hidden = model.init_hidden(1)
                    idx = corpus.dictionary.word2idx[self.first_word]
                    input = torch.tensor([[idx]], dtype=torch.long).to(device)
                    outf.write(self.first_word + ' ')

                    for i in range(self.words_per_sentence):
                        output, hidden = model(input, hidden)
                        word_weights = output.squeeze().div(self.temperature).exp().cpu()
                        word_idx = torch.multinomial(word_weights, 1)[0]
                        input.fill_(word_idx)

                        word = corpus.dictionary.idx2word[word_idx]
                        outf.write(word + ' ')
                        
                        if word == '.':
                            break;

                    outf.write('\n')

        print("Text created.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="./data/3/", type=str, metavar='N',
                        help='data path (default ./data/3/)')
    parser.add_argument('--model_file', default="./data/9/save.mdl", type=str, metavar='N',
                        help='model path (default ./data/9/save.mdl)')
    parser.add_argument('--output_path', default="./data/9/", type=str, metavar='N',
                        help='output path (default ./data/9/)')
    parser.add_argument('--num_sentences', default="10", type=int, metavar='N',
                        help='Number of produced sentences (default 10)')
    parser.add_argument('--first_word', default="This", type=str, metavar='N',
                        help='First word to generate(default "This")')
    args = parser.parse_args()

    net = Generator(args)
    net.run()
