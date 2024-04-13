# Project2 for EN.520.666 Information Extraction

# 2021 Ruizhe Huang
# 2022 Zili Huang

import numpy as np
import matplotlib.pyplot as plt
import string

import os
from HMM import HMM
from collections import Counter
from itertools import chain
from tqdm import tqdm
import pickle

NOISE = "<noise>"
data_dir = "data"

def read_file_line_by_line(file_name, func=lambda x: x, skip_header=True):
    print("reading file: %s" % file_name)
    res = list()
    with open(file_name, "r") as fin:
        if skip_header:
            fin.readline()  # skip the header
        for line in fin:
            if len(line.strip()) == 0:
                continue
            fields = func(line.strip())
            res.append(fields)
    print("%d lines, done" % len(res))
    return res


class Word_Recognizer:

    def __init__(self, restore_ith_epoch=None):
        # read labels
        self.lblnames = read_file_line_by_line(os.path.join(data_dir, "clsp.lblnames"))

        # read training data
        self.trnlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.trnlbls"), func=lambda x: x.split())
        self.endpts = read_file_line_by_line(os.path.join(data_dir, "clsp.endpts"), func=lambda x: list(map(int, x.split())))
        self.trnscr = read_file_line_by_line(os.path.join(data_dir, "clsp.trnscr"))

        # read dev data
        self.devlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.devlbls"), func=lambda x: x.split())
        self.train_words = set(self.trnscr)

        assert len(self.trnlbls) == len(self.endpts)
        assert len(self.trnlbls) == len(self.trnscr)

        # 23 letters + noise
        self.letters = list(string.ascii_lowercase)
        for c in ['k', 'q', 'z']:
            self.letters.remove(c)
        self.noise_id = len(self.letters)
        self.letters.append(NOISE)
        self.letter2id = dict({c: i for i, c in enumerate(self.letters)})
        self.id2letter = dict({i: c for c, i in self.letter2id.items()})

        # 256 quantized feature-vector labels
        self.label2id = dict({lbl: i for i, lbl in enumerate(self.lblnames)})
        self.id2label = dict({i: lbl for lbl, i in self.label2id.items()})

        # convert file contents to integer ids
        self.trnlbls = [[self.label2id[lbl] for lbl in line] for line in self.trnlbls]
        self.devlbls = [[self.label2id[lbl] for lbl in line] for line in self.devlbls]
        self.trnscr = [[self.letter2id[c] for c in word] for word in self.trnscr]

        # get label frequencies
        lbl_freq = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=0)
        lbl_freq_noise = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=1, endpts=self.endpts)

        # get hmms for each letter
        #self.letter_id2hmm = self.init_letter_hmm(lbl_freq, lbl_freq_noise, self.id2letter)

    def get_unigram(self, trnlbls, nlabels, smooth=0, endpts=None):
        # Compute "unigram" frequency of the training labels
        # Return freq(np array): the "unigram" frequency of the training labels
        freq = np.zeros(nlabels)
        # If end points are not specified
        if endpts == None:
            # Iterate through each line (sequence of labels) in the training data
            for line in trnlbls:
                # Iterate through each label in the line
                for label in line:
                    # Increment the frequency count for the current label
                    freq[label] += 1
        else:
            # If end points are specified
            # Iterate through each line (sequence of labels) in the training data
            for line, endpoints in zip(trnlbls, endpts):
                # Iterate through the labels in the range from the beginning to the start end point
                for label in line[:endpoints[0]+1]:
                    # Increment the frequency count for the current label
                    freq[label] += 1
                # Iterate through the labels in the range from the end end point to the end
                for label in line[endpoints[1]:]:
                    # Increment the frequency count for the current label
                    freq[label] += 1      
        # Add smoothing factor to the frequencies
        freq = freq + smooth
        # Normalize the frequencies to obtain probabilities
        freq = freq / np.sum(freq)
        # Return the computed unigram frequencies
        print("freq", freq)
        return freq


    def init_letter_hmm(self, lbl_freq, lbl_freq_noise, id2letter):
        # Initialize the HMM for each letter
        # Return letter_id2hmm(dict): the key is the letter_id and the value
        # is the corresponding HMM
        transition = np.asarray([[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 0.8]])
        transition_noise = np.asarray([[0.25, 0.25, 0.25, 0.25, 0.0], 
                                          [0.0, 0.25, 0.25, 0.25, 0.25], 
                                          [0.0, 0.25, 0.25, 0.25, 0.25], 
                                          [0.0, 0.25, 0.25, 0.25, 0.25], 
                                          [0.0, 0.0, 0.0, 0.0, 0.75]])
        emission = np.zeros((256, 3, 3))
        emission_noise = np.zeros((256, 5, 5))

        raise NotImplementedError 

        return letter_id2hmm

    def id2word(self, w):
        # w should be a list of char ids
        return ''.join(map((lambda c: self.id2letter[c]), w))

    def get_word_model(self, scr):
        # Construct the word HMM based on self.letter_id2hmm
        # Return h(HMM object): the word HMM for the word scr 
        raise NotImplementedError 
        return h

    def update_letter_counters(self, scr, word_hmm):
        # Update self.letter_id2hmm based on the counts from 
        # word_hmm
        raise NotImplementedError 

    def train(self, num_epochs=1):

        # sort trnlbls, endpts and trnscr such that the same word appear next to each other
        trnlbls_sorted = []
        trnscr_sorted = []
        for scr, lbls in sorted(zip(self.trnscr, self.trnlbls)):
            trnlbls_sorted.append(lbls)
            trnscr_sorted.append(scr)

        # training for this many epochs
        log_likelihood = 0
        num_frames = 0
        for i_epoch in range(num_epochs):
            print("---- echo: %d ----" % i_epoch)
            raise NotImplementedError 
            print("log_likelihood =", log_likelihood, "per_frame_log_likelihood =", log_likelihood / num_frames)

            self.save(i_epoch)
            self.test()

    def test(self):
        # Compute the word likelihood for each dev samples 
        id2words = dict({i: w for i, w in enumerate(self.train_words)})
        words2id = dict({w: i for i, w in id2words.items()})
        word_likelihoods = np.zeros((len(words2id), len(self.devlbls)))
        raise NotImplementedError 

        result = word_likelihoods.argmax(axis=0)
        result = [id2words[res] for res in result]

    def save(self, i_epoch):
        fn = os.path.join(data_dir, "%d.mdl.pkl" % i_epoch)
        print("Saved to:", fn)
        # for letter_id, hmm in self.letter_id2hmm.items():
            # hmm.output_arc_counts = None
            # hmm.output_arc_counts_null = None
        pickle.dump(self.letter_id2hmm, open(fn, "wb"))

    def load(self, i_epoch):
        return pickle.load(open(os.path.join(data_dir, "%d.mdl.pkl" % i_epoch), "rb"))


def main():
    n_epochs = 1
    wr = Word_Recognizer()
    wr.train(num_epochs=n_epochs)

if __name__ == '__main__':
    main()