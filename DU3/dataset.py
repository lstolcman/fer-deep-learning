from __future__ import print_function, division

import os
import sys
import re
import pdb
import time

import numpy as np
import scipy as sp
import tensorflow as tf


class Dataset:
    
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.batch_index = 0
    
    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        # count and sort most frequent characters
        chars, cnts = np.unique(list(data), return_index=True)
        self.sorted_chars = chars[np.argsort(-cnts)]
        self.vocab_size = len(self.sorted_chars)
        
        # other way
        #cntr = Counter(data)
        #self.sorted_chars = sorted(cntr.keys(), key=cntr.get, reverse=True)

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
        self.id2char = {k:v for v,k in self.char2id.items()}
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        return np.array([self.char2id[c] for c in sequence], dtype=np.int32)

    def decode(self, encoded_sequence):
        return [self.id2char[c] for c in encoded_sequence]
        
    def create_minibatches(self):
        data_len = len(self.x)
        chars_per_batch = self.batch_size * self.sequence_length
        self.num_batches = int((data_len-1) / chars_per_batch) 
 
        self.batches = np.zeros([self.num_batches, self.batch_size, self.sequence_length + 1], dtype=np.int32)      
        for b in range(self.num_batches):
            for s in range(self.batch_size):
                sentance_start = s*(self.num_batches*self.sequence_length)
                start = b * self.sequence_length + sentance_start
                end = start + self.sequence_length + 1 
                self.batches[b, s, :] = self.x[start:end]
                        
        self.batch_index = 0

    def next_minibatch(self):
        new_epoch = self.batch_index == self.num_batches
        if new_epoch:
            self.batch_index = 0

        batch = self.batches[self.batch_index, :, :]
        self.batch_index += 1
        
        batch_x = batch[:, :-1]
        batch_y = batch[:, 1:]
        return new_epoch, batch_x, batch_y
    
    def _as_one_hot(self, x, vocab):
        n = len(x)
        Yoh = np.zeros((n, vocab))
        Yoh[np.arange(n), x] = 1
        return Yoh
    

    def one_hot(self, batch):
        if batch.ndim == 1:
            return self._as_one_hot(batch, self.vocab_size)
        else:
            return np.array([self._as_one_hot(s, self.vocab_size) for s in batch])
