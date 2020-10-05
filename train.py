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


# Text model
class Trainer():
    
    def __init__(self, args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.output_path = args.output_path;

        self.prepare_batch = None
        self.corpus = None

        self.seed = 1111
        self.lr = 20
        self.eval_batch_size = 10
        self.bptt = 35
        self.emsize = 200
        self.nhid = 200
        self.nlayers = 2
        self.dropout = 0.2
        self.tied = True
        self.model_type = "LSTM"
        self.clip = 0.25
        self.log_interval = 200

        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.seed)

    # Load datasekt and split into training and test sets.
    def load_dataset(self):
        self.corpus = data.Corpus(self.data_path)

        train_data = self.batchify(self.corpus.train, self.batch_size)
        val_data = self.batchify(self.corpus.valid, self.eval_batch_size)
        test_data = self.batchify(self.corpus.test, self.eval_batch_size)

        return train_data, val_data, test_data;

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def create_model(self, device):
        ntokens = len(self.corpus.dictionary)
        net = model.RNNModel(self.model_type, ntokens, self.emsize, self.nhid, self.nlayers, self.dropout, self.tied).to(device)

        self.criterion = nn.NLLLoss()

        return net

    def train(self, model, train_data):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = len(self.corpus.dictionary)
        hidden = model.init_hidden(self.batch_size)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bptt)):
            data, targets = self.get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            hidden = self.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = self.criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-self.lr)

            total_loss += loss.item()

            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                print('| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                    batch, len(train_data) // self.bptt, self.lr,
                    elapsed * 1000 / self.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(self, model, data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = len(self.corpus.dictionary)
        hidden = model.init_hidden(self.eval_batch_size)

        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = self.get_batch(data_source, i)
                output, hidden = model(data, hidden)
                hidden = self.repackage_hidden(hidden)
                total_loss += len(data) * self.criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)
    
    def run(self):

        # GPU related settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch:", torch.__version__, "(CPU+GPU)" if  torch.cuda.is_available() else "(CPU)")
        torch.cuda.set_device(0)

        # Create directories
        if not os.path.isdir('data/9'):
            os.mkdir('data/9')

        # Load dataset
        train_data, val_data, test_data = self.load_dataset()

        # Loop over epochs.
        lr = self.lr
        best_val_loss = None

        # Create model
        model = self.create_model(device)

        for epoch in range(1, self.epochs+1):
            epoch_start_time = time.time()
            self.train(model, train_data)
            val_loss = self.evaluate(model, val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(self.output_path + "/save.mdl", 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

        # Load the best saved model.
        with open(self.output_path + "/save.mdl", 'rb') as f:
            model = torch.load(f)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            # Currently, only rnn model supports flatten_parameters function.
            model.rnn.flatten_parameters()

        # Run on test data.
        test_loss = self.evaluate(model, test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run (default 1)')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='batch size (default 32)')
    parser.add_argument('--data_path', default="./data/3/", type=str, metavar='N',
                        help='data path (default ./data/3/)')
    parser.add_argument('--ouput_path', default="./data/9/", type=str, metavar='N',
                        help='output path (default ./data/9/)')
    args = parser.parse_args()

    net = Trainer(args)
    net.run()
