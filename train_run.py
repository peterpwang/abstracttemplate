import train

import argparse
import sys
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run (default 1)')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='batch size (default 128)')
    args = parser.parse_args()

    model = train.Trainer(args)
    model.run()
