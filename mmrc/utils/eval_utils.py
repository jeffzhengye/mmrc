# encoding: utf-8

import pickle


class CVRecorder(object):
    def __init__(self, fold, acc, train_idx, val_idx, fold_num=10):
        self.fold_num = fold_num
        self.fold = fold
        self.acc = acc
        self.train_idx = train_idx
        self.val_idx = val_idx

    def save(self, output):
        output = output + ".pick"
        with open(output, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        filename += ".pick"
        with open(filename, 'rb') as f:
            return pickle.load(f)
