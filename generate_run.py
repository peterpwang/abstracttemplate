import generate

import argparse
import sys
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    model = generate.Generator(args)
    model.run()
