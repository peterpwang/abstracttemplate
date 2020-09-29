import csv
import sys
import os


def read_text(text_path):
    data = []
    with open(text_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for row in csv_reader:
            data.append(row[0])
    return data

