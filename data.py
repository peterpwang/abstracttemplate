import os
from io import open
import torch
from torchtext.datasets.language_modeling import LanguageModelingDataset
from torchtext.data.utils import get_tokenizer

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
        dataset = LanguageModelingDataset(path, TEXT);
        trainset, validationset, testset = dataset.split(split_ratio=(0.7,0.2,0.1))

        self.train = self.tokenize(trainset)
        self.valid = self.tokenize(validationset)
        self.test = self.tokenize(testset)

    def tokenize(self, dataset):
        """Tokenizes a dataset."""
        # Add words to the dictionary

        loader = list(torch.utils.data.DataLoader(dataset, num_workers=2))

        for line in loader:
            words = line.split() + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        idss = []
        for line in loader:
            words = line.split() + ['<eos>']
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

        return ids
