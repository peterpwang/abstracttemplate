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
        self.model_path = args.model_path

        self.corpus = None

        self.seed = 1111
        self.log_interval = 200
        self.generated_path = "./data/9/generated.txt"
        self.words = 100
        self.temperature = 1.0

        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.seed)

    def run(self):

        # GPU related settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch:", torch.__version__, "(CPU+GPU)" if  torch.cuda.is_available() else "(CPU)")
        torch.cuda.set_device(0)

        with open(self.model_path, 'rb') as f:
            model = torch.load(f).to(device)
        model.eval()

        print("Model read.")

        corpus = data.Corpus(self.data_path)
        ntokens = len(corpus.dictionary)

        hidden = model.init_hidden(1)
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

        with open(self.generated_path, 'w') as outf:
            with torch.no_grad():  # no tracking history
                for i in range(self.words):
                    output, hidden = model(input, hidden)
                    word_weights = output.squeeze().div(self.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)

                    word = corpus.dictionary.idx2word[word_idx]

                    outf.write(word + ('\n' if i % 20 == 19 else ' '))

                    if i % self.log_interval == 0:
                        print('| Generated {}/{} words'.format(i, self.words))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="./data/3/", type=str, metavar='N',
                        help='data path (default ./data/3/)')
    parser.add_argument('--model_path', default="./data/9/save.mdl", type=str, metavar='N',
                        help='model path (default ./data/9/save.mdl)')
    args = parser.parse_args()

    net = Generator(args)
    net.run()
