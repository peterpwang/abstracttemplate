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
    parser.add_argument('--data_path', default="./data/text/", type=str, metavar='N',
                        help='data path (default ./data/text/)')
    args = parser.parse_args()

    model = train.Trainer(args)
    model.run()
