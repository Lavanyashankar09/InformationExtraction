
# Project1 for EN.520.666 Information Extraction

# 2021 Matthew Ost
# 2021 Ruizhe Huang
# 2022 Zili Huang

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import string

EPS = 1e-6

def load_data(fname):
    alphabet_string = string.ascii_lowercase
    char_list = list(alphabet_string)
    print(char_list)
    char_list.append(' ')
    with open(fname, 'r') as fh:
        content = fh.readline()
    content = content.strip('\n')
    data = []
    for c in content:
        assert c in char_list
        data.append(char_list.index(c))
    return np.array(data) 

def get_init_prob_2states():
    # Define initial transition probability and emission probability
    # for 2 states HMM
    T_prob = np.array([[0.49, 0.51], [0.51, 0.49]])
    E_prob = np.zeros((2, 27))
    E_prob[0,0:13] = 0.0370
    E_prob[1,0:13] = 0.0371
    E_prob[0,13:26] = 0.0371
    E_prob[1,13:26] = 0.0370
    E_prob[0,26] = 0.0367
    E_prob[1,26] = 0.0367
    # row1_values = [0.0675, 0.0129, 0.0288, 0.0361, 0.1058, 0.0177, 0.0139, 0.0379, 0.0574, 0.0033, 
    #            0.0040, 0.0386, 0.0186, 0.0547, 0.0645, 0.0187, 0.0001, 0.0542, 0.0555, 0.0777, 
    #            0.0234, 0.0076, 0.0135, 0.0032, 0.0152, 0.0005, 0.1687]

    # # Values for the second row
    # row2_values = [0.0680, 0.0120, 0.0285, 0.0357, 0.1042, 0.0187, 0.0153, 0.0366, 0.0584, 0.0022, 
    #             0.0050, 0.0396, 0.0198, 0.0536, 0.0655, 0.0199, 0.0016, 0.0525, 0.0568, 0.0780, 
    #             0.0220, 0.0085, 0.0130, 0.0024, 0.0128, 0.0004, 0.1690]

    # # Creating numpy matrix
    # E_prob = np.array([row1_values, row2_values])
    # T_prob = np.array([[0.5, 0.5], [0.5, 0.5]])
    # E_prob = np.zeros((2, 27))
    # E_prob[:,:] = 1/27
    return T_prob, E_prob

def get_init_prob_4states(): 
    T_prob = np.array([[0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25]])
    E_prob = np.zeros((4, 27))
    E_prob[0, 0:13] =  0.0370
    E_prob[1, 13:27] =  0.0370
    E_prob[0, 13:27] =  0.0371
    E_prob[1, 0:13] =  0.0371
    E_prob[1, 26] = 0.0367
    E_prob[0, 26] = 0.0367
    E_prob[2, 0:13] =  0.0370
    E_prob[3, 13:27] =  0.0370
    E_prob[2, 13:27] =  0.0371
    E_prob[3, 0:13] =  0.0371
    E_prob[3, 26] = 0.0367
    E_prob[2, 26] = 0.0367
    return T_prob, E_prob

def read_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def calc_relative_freq(text, letters):
    letter_counts = {}
    for letter in letters:
        letter_counts[letter] = text.count(letter)
        
    total_chars = sum(letter_counts.values())
    relative_freqs = {}
    for letter, count in letter_counts.items():
        relative_freqs[letter] = count / total_chars
    
    return relative_freqs

def q3(filename, Y, lambda_val=0.002):
    Y_size = 27
    Y = "abcdefghijklmnopqrstuvwxyz "
    filename = "textA.txt"
    text_A = read_file(filename)
    rel_freq = calc_relative_freq(text_A, Y)
    r_y = np.random.rand(Y_size)
    mean_r_y = np.mean(r_y)
    delta_y = r_y - mean_r_y
    lambda_value = 0.003

    while lambda_value > 0.001 :
        q_y_minus = {}
        q_y_plus = {}
        i = 0
        for letter in Y:
            q_y_minus[letter] = rel_freq[letter] - lambda_value * delta_y[i]
            q_y_plus[letter] = rel_freq[letter] + lambda_value * delta_y[i]
            i += 1

        all_positive_minus = True
        all_positive_plus = True
        for q in q_y_minus.values():
            if q <= 0:
                all_positive_minus = False
                break
        for q in q_y_plus.values():
            if q <= 0:
                all_positive_plus = False
                break

        if all_positive_minus and all_positive_plus:
            break
        else:
            lambda_value -= 0.00001

    print("Chosen Lambda:", lambda_value)
    print("\nq values with lambda for q(y|1):")
    for letter, q_value in q_y_minus.items():
        print(f"Letter '{letter}': {q_value:.4f}")
    print("\nq values with lambda for q(y|2):")
    for letter, q_value in q_y_plus.items():
        print(f"Letter '{letter}': {q_value:.4f}")

class HMM:

    def __init__(self, num_states, num_outputs):
        # Args:
        #     num_states (int): number of HMM states
        #     num_outputs (int): number of output symbols            

        self.states = np.arange(num_states)  # just use all zero-based index
        self.outputs = np.arange(num_outputs)
        self.num_states = num_states
        self.num_outputs = num_outputs

        # Probability matrices
        self.transitions = None
        self.emissions = None

    def initialize(self, T_prob, E_prob):
        # Initialize HMM with transition probability T_prob and emission probability
        # E_prob

        # Args:
        #     T_prob (numpy.ndarray): [num_states x num_states] numpy array.
        #     T_prob[i, j] is the transition probability from state i to state j.
        #     E_prob (numpy.ndarray): [num_states x num_outputs] numpy array.
        #     E_prob[i, j] is the emission probability of state i to output jth symbol. 
        self.transitions = T_prob
        self.emissions = E_prob
        self._assert_transition_probs()
        self._assert_emission_probs()

    def _assert_emission_probs(self):
        for s in self.states:
            assert self.emissions[s].sum() - 1 < EPS

    def _assert_transition_probs(self):
        for s in self.states:
            assert self.transitions[s].sum() - 1 < EPS
            assert self.transitions[:, s].sum() - 1 < EPS
    
    def forward(self, Obs_seq_y):
        if self.num_states == 2:
            Initial = np.array([0.5, 0.5])
        else:
            Initial = np.array([0.25, 0.25, 0.25, 0.25])
        
        alpha = np.zeros((self.num_states,len(Obs_seq_y)))
        #Initialization
        alpha[:,0] = Initial * self.emissions[:, Obs_seq_y[0]]
        norm = [1]
        #norm.append(np.sum(alpha[:, 0]))  
        #Induction
        for i in range(1, len(Obs_seq_y)):
            for yi in range(self.num_states):
                for yi_1 in range(self.num_states):
                    alpha[yi, i] += alpha[yi_1, i-1] * self.transitions[yi_1, yi] * self.emissions[yi, Obs_seq_y[i]]
            #Normalization
           
            norm_i = np.sum(alpha[:, i]) 
            norm.append(norm_i)
            alpha[:, i] = alpha[:, i] / norm_i
        return alpha, norm

    
    def backward(self, Obs_seq_y,norm):
        beta = np.zeros((self.num_states, len(Obs_seq_y)))
        #Initialization
        beta[:,-1] = 1
        #Induction
        for i in range(len(Obs_seq_y)-2, -1, -1):
            #print(len(Obs_seq_y)-2)
            for yi in range(self.num_states):
                for yi_1 in range(self.num_states):
                    beta[yi, i] += beta[yi_1, i+1] * self.transitions[yi, yi_1] * self.emissions[yi_1, Obs_seq_y[i+1]]
            #Normalization
            beta[:, i] = beta[:, i] / norm[i+1]
        return beta

    def Baum_Welch(self, max_iter, train_data, test_data):
        # The Baum Welch algorithm to estimate HMM parameters
        # Args:
        #     max_iter (int): maximum number of iterations to train
        #     train_data (numpy.ndarray): train data
        #     test_data (numpy.ndarray): test data
        #
        # Returns:
        #     info_dict (dict): dictionary containing information to visualize

        info_dict = {'iteration' : [],
                    'train_log_likelihood' : [],
                    'test_log_likelihood' : [],
                    'emi_a': [[],[],[],[]],
                    'emi_n': [[],[],[],[]]}

        for i in range(max_iter):
            # Implement the Baum-Welch algorithm here

            # The forward pass
            alpha, norm = self.forward(train_data)
            # The backward pass
            beta = self.backward(train_data, norm)

            info_dict['iteration'].append(i)
            info_dict['train_log_likelihood'].append(self.log_likelihood(train_data))
            info_dict['test_log_likelihood'].append(self.log_likelihood(test_data))
            
            for j in range(self.num_states):
                info_dict['emi_a'][j].append(self.emissions[j, 0])
                info_dict['emi_n'][j].append(self.emissions[j, 13])

            print("iteration",i)
            print("train",info_dict['train_log_likelihood'][-1])
            print("test",info_dict['test_log_likelihood'][-1])
            
            # print(self.transitions)
            # print(self.emissions)
            # Transition probability evaluation
            
            counts = np.zeros((self.num_states, self.num_states))
            for i in range(len(train_data)-1):
                for yi in range(self.num_states):
                    for yi_1 in range(self.num_states):
                        counts[yi, yi_1] += alpha[yi, i] * self.transitions[yi, yi_1] * self.emissions[yi_1, train_data[i+1]] * beta[yi_1, i+1] / norm[i+1]
            
            #print("counts",counts)
            
            for yi in range(self.num_states):
                for yi_1 in range(self.num_states):
                    self.transitions[yi, yi_1] = counts[yi, yi_1] / np.sum(counts[yi, :])
            
            
            gamma = np.multiply(alpha, beta) 
            # print("gamma",gamma)
            # print("gamma shape",gamma.shape)
            for yi in range(self.num_states):
                for k in range(len(self.emissions[0])):
                    #print("k",k)
                    #print("gamma[yi, train == k]",gamma[yi, train == k])
                    # print("len(gamma[yi, train == k])",len(gamma[yi, train == k]))
                    # print("len(gamma[yi, :])",len(gamma[yi, :]))
                    self.emissions[yi, k] = np.sum(gamma[yi, train_data == k]) / np.sum(gamma[yi, :])
        return info_dict

    def log_likelihood(self, data):
        # Compute the log likelihood of sequence data
        # Args:
        #     data (numpy.ndarray): 
        #
        # Returns:
        #     prob (float): log likelihood of data
        alpha, norm = self.forward(data)
        prob = np.sum(np.log(norm))/len(data)
        return prob

    def visualize(self, info_dict):
    # Plot average log-probability of training and test data
        plt.figure(figsize=(10, 5))
        plt.plot(info_dict['iteration'], info_dict['train_log_likelihood'], label='Train Data')
        plt.plot(info_dict['iteration'], info_dict['test_log_likelihood'], label='Test Data')
        plt.xlabel('Iterations')
        plt.ylabel('Average Log-Probability')
        plt.title('Average Log-Probability of Training and Test Data')
        plt.legend()
        plt.grid(True)
        plt.savefig('average_log_probability.png')  # Save the plot as an image
        #plt.show()

        # Plot emission probabilities of particular letters for each state
        plt.figure(figsize=(10, 5))
        plt.plot(info_dict['iteration'], info_dict['emi_a'][0], label='emmission of a|1')
        plt.plot(info_dict['iteration'], info_dict['emi_a'][1], label='emmission of a|2') 
        plt.plot(info_dict['iteration'], info_dict['emi_n'][0], label='emmission of n|1')
        plt.plot(info_dict['iteration'], info_dict['emi_n'][1], label='emmission of n|2')
        plt.xlabel('Iterations')
        plt.ylabel('Emission Probabilities')
        plt.title('Emission Probabilities of particular letters for each state')
        plt.legend()
        plt.grid(True)
        plt.savefig('emission_probabilities.png')  # Save the plot as an image
        #plt.show()

def main():
    n_states = 4
    n_outputs = 27
    train_file, test_file = "textA.txt", "textB.txt"
    max_iter = 2
    # define initial transition probability and emission probability
    #T_prob, E_prob = get_init_prob_2states() 
    T_prob, E_prob = get_init_prob_4states() 
    ## initial the HMM class
    H = HMM(num_states=n_states, num_outputs=n_outputs)
    ## initialize HMM with the transition probability and emission probability
    H.initialize(T_prob, E_prob)
    # load text file
    train_data, test_data = load_data(train_file), load_data(test_file)
    ## train the parameters of HMM
    info_dict = H.Baum_Welch(max_iter, train_data, test_data)
    # ## visualize
    H.visualize(info_dict)
    # q3("textA.txt", Y)

if __name__ == "__main__":
    main()