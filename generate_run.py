import generate

import argparse
import sys
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="./data/text/", type=str, metavar='N',
                        help='data path (default ./data/text/)')
    parser.add_argument('--model_path', default="./data/results/save.mdl", type=str, metavar='N',
                        help='model path (default ./data/results/save.mdl)')
    args = parser.parse_args()

    model = generate.Generator(args)
    model.run()
