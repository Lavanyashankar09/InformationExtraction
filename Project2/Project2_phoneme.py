
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
from scipy.special import logsumexp

NOISE = "<noise>"
data_dir = "data/"
# Function to read file line by line
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

# Class for Word Recognizer
class Word_Recognizer:

    def __init__(self, restore_ith_epoch=None):
        # Read data files
        self.lblnames = read_file_line_by_line(os.path.join(data_dir, "clsp.lblnames"))
        self.trnlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.trnlbls"), func=lambda x: x.split())
        self.endpts = read_file_line_by_line(os.path.join(data_dir, "clsp.endpts"), func=lambda x: list(map(int, x.split())))
        self.trnscr = read_file_line_by_line(os.path.join(data_dir, "clsp.trnscr"))
        self.devlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.devlbls"), func=lambda x: x.split())
        self.train_words = set(self.trnscr)
        self.train_words_sorted = sorted(self.train_words)

        assert len(self.trnlbls) == len(self.endpts)
        assert len(self.trnlbls) == len(self.trnscr)

        # Define letters and noise
        self.letters = list(string.ascii_lowercase)
        for c in ['k', 'q', 'z']:
            self.letters.remove(c)
        self.noise_id = len(self.letters)
        self.letters.append(NOISE)


        self.phonemes = ["AA", "AE", "AH", "AO", "AW", "AY","B", "CH", "D", "DH", "EH", "ER", "EY","F", "G", "IH", "IY", "JH","K", "L", "M", "N", "NG", "OW", "OY","P", "R", "S", "SH", "T", "TH", "UW", "V", "W", "Z"]
        self.noise_id_phonemes = len(self.phonemes)
        self.phonemes.append(NOISE)

        with open("data/clsp.trnphoneme", "r") as f:
            pp = [line.strip()[:] for line in f]
        self.dict_p = dict(zip(self.train_words_sorted, pp))

        # Mapping between letters and IDs
        self.letter2id = dict({c: i for i, c in enumerate(self.letters)})
        self.id2letter = dict({i: c for c, i in self.letter2id.items()})
        self.phonemes2id = dict({c: i for i, c in enumerate(self.phonemes)})
        self.id2phonemes = dict({i: c for c, i in self.phonemes2id.items()})


        # Mapping between labels and IDs
        self.label2id = dict({lbl: i for i, lbl in enumerate(self.lblnames)})
        self.id2label = dict({i: lbl for lbl, i in self.label2id.items()})

        # Convert file contents to integer IDs
        self.trnlbls = [[self.label2id[lbl] for lbl in line] for line in self.trnlbls]
        self.devlbls = [[self.label2id[lbl] for lbl in line] for line in self.devlbls]
        self.trnscr = [[self.letter2id[c] for c in word] for word in self.trnscr]

        # Get label frequencies
        lbl_freq = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=0)
        lbl_freq_noise = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=1, endpts=self.endpts)

        # Initialize HMMs for each letter
        self.phonemes_id2hmm = self.init_p_hmm(lbl_freq, lbl_freq_noise)
        self.log_likelihoods = []

    def get_unigram(self, trnlbls, nlabels, smooth=0, endpts=None):
        # Compute "unigram" frequency of the training labels
        unigram_freq = np.zeros(nlabels)
        if endpts == None:
            for line in trnlbls:
                for label in line:
                    unigram_freq[label] += 1
        else:
            for line, endpoints in zip(trnlbls, endpts):
                for label in line[:endpoints[0]+1]:
                    unigram_freq[label] += 1
                for label in line[endpoints[1]:]:
                    unigram_freq[label] += 1
        unigram_freq = unigram_freq + smooth
        unigram_freq = unigram_freq / np.sum(unigram_freq)
        return unigram_freq

    def init_p_hmm(self, lbl_freq, lbl_freq_noise):
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

        pid2hmm = dict()
        for phonemes_id in range(len(self.phonemes)):
            if phonemes_id != self.noise_id_phonemes:
                num_states = 3
                transition1 = transition
                emission1 = emission
            else:
                num_states = 5
                transition1 = transition_noise
                emission1 = emission_noise
            
            # Create an HMM object and initialize it with the appropriate parameters
            phmm = HMM(num_states=num_states, num_outputs=256)
            phmm.init_transition_probs(transition1)
            phmm.init_emission_probs(emission1)
            # Add the HMM to the dictionary with the letter_id as the key
            pid2hmm[phonemes_id] = phmm

        return pid2hmm


    def get_word_model(self, scr):
        # Construct the word HMM based on self.letter_id2hmm
        # Return h(HMM object): the word HMM for the word scr 
        id_to_word = ''.join(map((lambda c: self.id2letter[c]), scr))
        phoneme_of_word = self.dict_p[id_to_word]
        pp = dict({c: i for i, c in enumerate(self.phonemes)})
        phonemes = phoneme_of_word.split()
        # Convert each phoneme to its corresponding ID
        last = [pp[phoneme] for phoneme in phonemes]
        noise_states = 5
        total_noise_states = noise_states*2
        word_states = 3 * len(last) 
        total_states = total_noise_states + word_states
        word_phonemes_HMM = HMM(num_states= total_states , num_outputs=256)
        transitions_word = np.zeros((total_states, total_states))
        emissions_word = np.zeros((256, total_states, total_states))
        transitions_word[:noise_states, :noise_states] = self.phonemes_id2hmm[self.noise_id_phonemes].transitions
        emissions_word[:, :noise_states, :noise_states] = self.phonemes_id2hmm[self.noise_id_phonemes].emissions
        transitions_word[-noise_states:, -noise_states:] = self.phonemes_id2hmm[self.noise_id_phonemes].transitions
        emissions_word[:, -noise_states:, -noise_states:] = self.phonemes_id2hmm[self.noise_id_phonemes].emissions
        
        null_arcs_word = defaultdict(dict)
        null_arcs_word[4][5] = 0.25
        for i, phoneme_id in enumerate(last):
            transitions_word[5+i*3:8+i*3, 5+i*3:8+i*3] = self.phonemes_id2hmm[phoneme_id].transitions
            emissions_word[:, 5+i*3:8+i*3, 5+i*3:8+i*3] = self.phonemes_id2hmm[phoneme_id].emissions
            null_arcs_word[4+(i+1)*3][5+(i+1)*3] = 0.2
        word_phonemes_HMM.init_transition_probs(transitions_word)
        word_phonemes_HMM.init_emission_probs(emissions_word)
        word_phonemes_HMM.init_null_arcs(null_arcs_word)
        return word_phonemes_HMM

    def update_letter_counters(self, scr, word_hmm):
        
        id_to_word = ''.join(map((lambda c: self.id2letter[c]), scr))
        phoneme_of_word = self.dict_p[id_to_word]
        p = dict({c: i for i, c in enumerate(self.phonemes)})
        phonemes = phoneme_of_word.split()
        last = [p[phoneme] for phoneme in phonemes]
        self.phonemes_id2hmm[self.noise_id_phonemes].set_counters(word_hmm.output_arc_counts[:, :5, :5], word_hmm.output_arc_counts_null)
        self.phonemes_id2hmm[self.noise_id_phonemes].set_counters(word_hmm.output_arc_counts[:, -5:, -5:], word_hmm.output_arc_counts_null)
        for i, iid in enumerate(last):
            self.phonemes_id2hmm[iid].set_counters(word_hmm.output_arc_counts[:, 5+i*3:8+i*3, 5+i*3:8+i*3], word_hmm.output_arc_counts_null)

    
    def train(self, num_epochs=1):
        # sort trnlbls, endpts and trnscr such that the same word appear next to each other
        
        trnlbls_sorted = []
        trnscr_sorted = []
        trnlbls_sorted_2 = []
        trnscr_sorted_2 = []
        
        # Split data into training and held-out sets
        split = int(len(self.trnscr) * 0.10)
        trainscr2 = self.trnscr[:split]
        trainlbls2 = self.trnlbls[:split]
        self.trainscr = self.trnscr[split:]
        self.trainlbls = self.trnlbls[split:]

        # Sort held-out set based on scores
        for score, label in sorted(zip(trainscr2, trainlbls2)):
            trnlbls_sorted_2.append(label)
            trnscr_sorted_2.append(score)

        # Sort training set based on scores
        for score, label in sorted(zip(self.trnscr, self.trnlbls)):
            trnlbls_sorted.append(label)
            trnscr_sorted.append(score)

        for i_epoch in range(num_epochs):
            log_likelihood = 0
            num_frames = 0
            for phoneme_id in self.phonemes_id2hmm:
                self.phonemes_id2hmm[phoneme_id].reset_counters()
            print("---- epoch: %d ----" % i_epoch)
           
            # iterate over each word in the training set
            for scr, lbls in zip(trnscr_sorted_2, trnlbls_sorted_2):
                word_phoneme_hmm = self.get_word_model(scr)
                init_prob = np.zeros((word_phoneme_hmm.num_states), dtype=np.float64)
                init_prob[0] = 1.0
                alpha, beta, q = word_phoneme_hmm.forward_backward(lbls, init_prob=init_prob, update_params=False)
                self.update_letter_counters(scr, word_phoneme_hmm)
                
                init_beta = np.asarray([1] * word_phoneme_hmm.num_states)
                log_likelihood += word_phoneme_hmm.compute_log_likelihood(lbls, init_prob=init_prob, init_beta=init_beta)
                num_frames += len(lbls)
            
            
            for phoneme_id in self.phonemes_id2hmm:             
                self.phonemes_id2hmm[phoneme_id].update_params()
            for phoneme_id in range(len(self.phonemes)):
                self.phonemes_id2hmm[phoneme_id].reset_counters()
            
            print("log_likelihood =", log_likelihood, "per_frame_log_likelihood =", log_likelihood / num_frames, )
            self.test(trnlbls_sorted_2,i_epoch)
            self.log_likelihoods.append(log_likelihood)
        return i_epoch


    def test(self, test_lbls, epoch_no):
        # Compute the word likelihood for each dev samples
        id2words = dict({i: w for i, w in enumerate(self.train_words)})
        words2id = dict({w: i for i, w in id2words.items()})

        for i, lbls in enumerate(test_lbls):
            word_likelihoods = np.zeros((len(words2id), len(test_lbls)))
            for j, word in enumerate(self.train_words):
                scr = []
                for letter in word:
                    scr.append(self.letter2id[letter])
                word_hmm = self.get_word_model(scr)
                init_prob = np.zeros((word_hmm.num_states))
                init_prob[0] = 1.0
                init_beta = np.ones((word_hmm.num_states))
                word_likelihoods[j, i] = word_hmm.compute_log_likelihood(lbls, init_prob=init_prob, init_beta=init_beta)
   
        
        results = word_likelihoods.argmax(axis=0)
        predicted_words = []
        for index in results:
            predicted_words.append(id2words[index])

        
        # Calculate the maximum likelihood score for each prediction
        max_likelihood_scores = word_likelihoods.max(axis=0)

        # Calculate the logarithm of the sum of exponentials of likelihood scores
        log_sum_exp_scores = logsumexp(word_likelihoods, axis=0)

        # Calculate the confidence scores for each prediction
        confidence = np.exp(max_likelihood_scores - log_sum_exp_scores)

        # Replace any NaN confidence scores with 1.0
        confidence[np.isnan(confidence)] = 1.0
        
        # print("Predicted Words: ", predicted_words)
        # print("Confidence: ", confidence)

         
    
    def plot_log_likelihood(self):
        # Plot log likelihood vs epoch
        print(self.log_likelihoods)

        epochs = range(1, len(self.log_likelihoods) + 1)
        print(epochs)
        plt.plot(epochs, self.log_likelihoods, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood vs Epoch')
        plt.grid(True)
        plt.grid(True)
        plt.savefig("log_likelihood_for_pppp.png")
        plt.show()


def main():
    n_epochs = 10
    wr = Word_Recognizer()
    wr.train(num_epochs=n_epochs)
    wr.plot_log_likelihood()
    

if __name__ == '__main__':
    main()
