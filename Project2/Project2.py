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
from collections import defaultdict

NOISE = "<noise>"
data_dir = "data/"

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
        #print("lblnames:", self.lblnames)
        # print("lblnames:", self.lblnames[0])
        # print("len(lblnames):", len(self.lblnames))
        # print("type(lblnames):", type(self.lblnames))
        # read training data
        self.trnlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.trnlbls"), func=lambda x: x.split())
        # print("trnlbls:", self.trnlbls[0])
        self.endpts = read_file_line_by_line(os.path.join(data_dir, "clsp.endpts"), func=lambda x: list(map(int, x.split())))
        # print("endpts:", self.endpts[0])
        # print("endpts[0]:", self.endpts[0][0])
        # print("type(endpts):", type(self.endpts))
        self.trnscr = read_file_line_by_line(os.path.join(data_dir, "clsp.trnscr"))
        # print("trnscr:", self.trnscr[0])
        # read dev data
        self.devlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.devlbls"), func=lambda x: x.split())
        self.train_words = set(self.trnscr)
        # print("train_words:", self.train_words)
        # print("len(train_words):", len(self.train_words)) 
        ## ==== 48

        assert len(self.trnlbls) == len(self.endpts)
        assert len(self.trnlbls) == len(self.trnscr)

        # 23 letters + noise
        self.letters = list(string.ascii_lowercase)
        for c in ['k', 'q', 'z']:
            self.letters.remove(c)
        self.noise_id = len(self.letters)
        self.letters.append(NOISE)
        #print("letters:", self.letters)
        self.letter2id = dict({c: i for i, c in enumerate(self.letters)})
        #print("letter2id:", self.letter2id)
        self.id2letter = dict({i: c for c, i in self.letter2id.items()})
        #print("id2letter:", self.id2letter)
        # 256 quantized feature-vector labels
        self.label2id = dict({lbl: i for i, lbl in enumerate(self.lblnames)})
        #key, value = next(iter( self.label2id.items()))
        # Printing the key-value pair
        #print(f"'{key}': '{value}'")
        self.id2label = dict({i: lbl for lbl, i in self.label2id.items()})
        #print("id2label:", self.id2label)
        #key, value = next(iter( self.id2label.items()))
        # Printing the key-value pair
        #print(f"'{key}': '{value}'")

        # convert file contents to integer ids
        self.trnlbls = [[self.label2id[lbl] for lbl in line] for line in self.trnlbls]
        #print("trnlbls:", self.trnlbls[0])
        self.devlbls = [[self.label2id[lbl] for lbl in line] for line in self.devlbls]
        self.trnscr = [[self.letter2id[c] for c in word] for word in self.trnscr]
        #print("trnscr:", self.trnscr[0])

        # get label frequencies
        lbl_freq = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=0)
        # print("self.lblnames:", self.lblnames[1])
        # print("len(self.lblnames):", len(self.lblnames))
        # print("self.trnlbls:", self.trnlbls[1])
        # print("lbl_freq:", lbl_freq[1])
        # print("lbl_freq.shape:", lbl_freq.shape)
        # print("type(lbl_freq):", type(lbl_freq))
        lbl_freq_noise = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=1, endpts=self.endpts)
        # print("lbl_freq_noise:", lbl_freq_noise[1])
        # print("lbl_freq_noise.shape:", lbl_freq_noise.shape)
        # get hmms for each letter
        self.letter_id2hmm = self.init_letter_hmm(lbl_freq, lbl_freq_noise, self.id2letter)
        # print("letter_id2hmm:", self.letter_id2hmm)
        # {0: <HMM.HMM object at 0x126b17d50>, 1: <HMM.HMM object at 0x126b17f10>, 2: <HMM.HMM object at 0x126b18350>,
        # print("value of letter_id2hmm:", self.letter_id2hmm[0])
        # Transition Probabilities: (3, 3)
        # [[0.8 0.2 0. ]
        # [0.  0.8 0.2]
        # [0.  0.  0.8]]
        # Emission Probabilities: (256, 3, 3)
        # [[[0.00281875 0.00281875 0.00281875]
        # [0.00281875 0.00281875 0.00281875]
        # [0.00281875 0.00281875 0.00281875]]

        # [[0.00230369 0.00230369 0.00230369]
        # [0.00230369 0.00230369 0.00230369]
        # [0.00230369 0.00230369 0.00230369]]
        # print("len(letter_id2hmm):", len(self.letter_id2hmm))
        # print("type(letter_id2hmm):", type(self.letter_id2hmm))
        self.log_likelihoods = []
    

    def get_unigram(self, trnlbls, nlabels, smooth=0, endpts=None):
        # Compute "unigram" frequency of the training labels
        # Return freq(np array): the "unigram" frequency of the training labels
        freq = np.zeros(nlabels)
        #print("type of freq ----", type(freq))
        # print("freq.shape", freq.shape)
        # If end points are not specified
        if endpts == None:
            for line in trnlbls:
                #print("len(line)", len(line))
                for label in line:
                    freq[label] += 1
                    #print("label", label)
            #print("freq", freq)
        else:
            for line, endpoints in zip(trnlbls, endpts):
                for label in line[:endpoints[0]+1]:
                    freq[label] += 1
                for label in line[endpoints[1]:]:
                    freq[label] += 1   
        
        #print("freq before", freq[1])
        freq = freq + smooth
        #print("freq", freq[1])
        freq = freq / np.sum(freq)
        #print("freq after", freq[1])
        # print("####################################")
        # print("freq", freq)
        # print("first 10 labels", freq[:10])
        # print("type(freq)", type(freq))
        # print("len(freq)", len(freq))
        # print("freq.shape", freq.shape)

        # print("####################################")
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
        for i in range(256):
            emission[i,:,:] = lbl_freq[i]
            emission_noise[i,:,:] = lbl_freq_noise[i]

        # print("emission.shape", emission.shape)
        # print("emission_noise.shape", emission_noise.shape)
        # print("emission", emission[1])
        # print("emission", emission[2])
        # print("emission_noise", emission_noise[1])
        # print("emission_noise", emission_noise[2])

        letter_id2hmm = dict()
        for letter_id in range(len(self.letters)):
        # Determine the number of states and transition/emission probabilities based on whether it's a noise letter or not
            if letter_id != self.noise_id:
                # print("letter_id", letter_id)
                # print("self.noise_id", self.noise_id)
                num_states = 3
                transition1 = transition
                emission1 = emission
            else:
                num_states = 5
                transition1 = transition_noise
                emission1 = emission_noise
            
            # Create an HMM object and initialize it with the appropriate parameters
            letter_hmm = HMM(num_states=num_states, num_outputs=256)
            letter_hmm.init_transition_probs(transition1)
            letter_hmm.init_emission_probs(emission1)
            #print("letter_hmm is ---------", letter_hmm)
            #print("letter_id is ---------", letter_id)
            
            # Add the HMM to the dictionary with the letter_id as the key
            letter_id2hmm[letter_id] = letter_hmm

        return letter_id2hmm

    def id2word(self, w):
        # w should be a list of char ids
        return ''.join(map((lambda c: self.id2letter[c]), w))

    def get_word_model(self, scr):
        # Construct the word HMM based on self.letter_id2hmm
        # Return h(HMM object): the word HMM for the word scr 
        noise_states = 5
        total_noise_states = noise_states*2
        word_states = 3 * len(scr) 
        total_states = total_noise_states + word_states
        h = HMM(num_states= total_states , num_outputs=256)
        transitions_word = np.zeros((total_states, total_states))
        emissions_word = np.zeros((256, total_states, total_states))
        transitions_word[:noise_states, :noise_states] = self.letter_id2hmm[self.noise_id].transitions
        emissions_word[:, :noise_states, :noise_states] = self.letter_id2hmm[self.noise_id].emissions
        transitions_word[-noise_states:, -noise_states:] = self.letter_id2hmm[self.noise_id].transitions
        emissions_word[:, -noise_states:, -noise_states:] = self.letter_id2hmm[self.noise_id].emissions
        
        null_arcs_word = defaultdict(dict)
        null_arcs_word[4][5] = 0.25
        for i, letter_id in enumerate(scr):
            transitions_word[5+i*3:8+i*3, 5+i*3:8+i*3] = self.letter_id2hmm[letter_id].transitions
            emissions_word[:, 5+i*3:8+i*3, 5+i*3:8+i*3] = self.letter_id2hmm[letter_id].emissions
            null_arcs_word[4+(i+1)*3][5+(i+1)*3] = 0.2

        h.init_transition_probs(transitions_word)
        h.init_emission_probs(emissions_word)
        h.init_null_arcs(null_arcs_word)

        #print("h is ---------", h)
        return h

    def update_letter_counters(self, scr, word_hmm):
        # Update self.letter_id2hmm based on the counts from 
        # word_hmm
        self.letter_id2hmm[self.noise_id].set_counters(word_hmm.output_arc_counts[:, :5, :5], word_hmm.output_arc_counts_null)
        self.letter_id2hmm[self.noise_id].set_counters(word_hmm.output_arc_counts[:, -5:, -5:], word_hmm.output_arc_counts_null)    
        for i, letter_id in enumerate(scr):
            self.letter_id2hmm[letter_id].set_counters(word_hmm.output_arc_counts[:, 5+i*3:8+i*3, 5+i*3:8+i*3],word_hmm.output_arc_counts_null)

    def train(self, num_epochs=1):
        # sort trnlbls, endpts and trnscr such that the same word appear next to each other
        trnlbls_sorted = []
        trnscr_sorted = []
        # print("trnlbls:", self.trnlbls[0])
        # print("trnscr:", self.trnscr[0])
        for scr, lbls in sorted(zip(self.trnscr, self.trnlbls)):
            trnlbls_sorted.append(lbls)
            trnscr_sorted.append(scr)

        # print("trnlbls_sorted:", trnlbls_sorted[0])
        # print("-------------------")
        # print("trnlbls_sorted:", trnlbls_sorted[1])
        # print("-------------------")
        # print("trnscr_sorted:", trnscr_sorted[0])
        # print("trnscr_sorted:", trnscr_sorted[1])
        # training for this many epochs
        
        for i_epoch in range(num_epochs):
            log_likelihood = 0
            num_frames = 0

            for letter_id in self.letter_id2hmm:
                self.letter_id2hmm[letter_id].reset_counters()
            
            print("---- echo: %d ----" % i_epoch)

            # iterate over each word in the training set
            for scr, lbls in zip(trnscr_sorted, trnlbls_sorted):
                # print("scr is ---------", scr)
                # print("lbls is ---------", lbls)
                word_hmm = self.get_word_model(scr)
                # print("word_hmm is ---------", word_hmm)
                # print("word_hmm.num_states", word_hmm.num_states)
                init_prob = np.zeros((word_hmm.num_states), dtype=np.float64)
                init_prob[0] = 1.0
                alpha, beta, q = word_hmm.forward_backward(lbls, init_prob=init_prob, update_params=False)
                self.update_letter_counters(scr, word_hmm)
                
                init_beta=np.asarray([1]*word_hmm.num_states)
                log_likelihood += word_hmm.compute_log_likelihood(lbls, init_prob=init_prob, init_beta=init_beta)

                num_frames += len(lbls)

            # update the parameters of the HMMs
            for letter_id in self.letter_id2hmm:
               self.letter_id2hmm[letter_id].update_params()
            for letter_id in range(len(self.letters)):
                self.letter_id2hmm[letter_id].reset_counters()
            print("log_likelihood =", log_likelihood, "per_frame_log_likelihood =", log_likelihood / num_frames)
            
            self.log_likelihoods.append(log_likelihood)

            self.save(i_epoch)
            self.test()
        

    def test(self):
        # Compute the word likelihood for each dev samples 
        # just devlabls
        id2words = dict({i: w for i, w in enumerate(self.train_words)})
        words2id = dict({w: i for i, w in id2words.items()})

        word_likelihoods = np.zeros((len(words2id), len(self.devlbls)))
        # print("word_likelihoods.shape:", word_likelihoods.shape)
        #(48, 393)

        for j, word in enumerate(self.train_words):
            scr = []
            for letter in word:
                scr.append(self.letter2id[letter])
            word_hmm = self.get_word_model(scr)
            init_prob = np.zeros((word_hmm.num_states))
            init_prob[0] = 1.0
            init_beta=np.ones((word_hmm.num_states))
                              
            for i, lbls in enumerate(self.devlbls):
                word_likelihoods[j, i] = word_hmm.compute_log_likelihood(lbls, init_prob=init_prob, init_beta=init_beta)

        result = word_likelihoods.argmax(axis=0)
        result = [id2words[res] for res in result]
        # print("result:", result)

    def save(self, i_epoch):
        fn = os.path.join(data_dir, "%d.mdl.pkl" % i_epoch)
        print("Saved to:", fn)
        for letter_id, hmm in self.letter_id2hmm.items():
            hmm.output_arc_counts = None
            hmm.output_arc_counts_null = None
        pickle.dump(self.letter_id2hmm, open(fn, "wb"))

    def load(self, i_epoch):
        return pickle.load(open(os.path.join(data_dir, "%d.mdl.pkl" % i_epoch), "rb"))
   
    def plot_log_likelihood(self):
        epochs = range(1, len(self.log_likelihoods) + 1)
        plt.plot(epochs, self.log_likelihoods, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood vs Epoch')
        plt.grid(True)
        plt.grid(True)
        plt.savefig("ll.png")
        

def main():
    n_epochs = 10
    wr = Word_Recognizer()
    wr.train(num_epochs=n_epochs)
    wr.plot_log_likelihood()

if __name__ == '__main__':
    main()