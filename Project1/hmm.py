import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    #Its output sequence 
    #Observation Sequence
    # Read the content of the file
    with open(filename, 'r') as file:
        content = file.read()
    # Convert each lowercase alphabet and space to its numeric value
    numeric_values = [(ord(char.lower()) - ord('a')) if char.isalpha() else 26 for char in content]
    Obs_seq_y = np.array(numeric_values)
    # Display the result
    # print(Obs_seq_y)
    # print(type(Obs_seq_y))
    # print(Obs_seq_y.shape)
    return Obs_seq_y

def get_init_prob_2states():
    #Start
    #its same for both
    Initial = np.array([0.5, 0.5])
    #its Hidden
    #Transition
    Transition_p = np.array([[0.49, 0.51], [0.51, 0.49]])
    # print(Transition_p)
    # print(Transition_p.shape)
    #its Observation
    #Emission
    Emission_q = np.zeros((2, 27))
    Emission_q[0,0:13] = 0.0370
    Emission_q[1,0:12] = 0.0371
    Emission_q[0,13:26] = 0.0371
    Emission_q[1,13:26] = 0.0370
    Emission_q[0,26] = 0.0367
    Emission_q[1,26] = 0.0367
    # print(Emission_q)
    # print(Emission_q.shape)
    return Initial, Transition_p, Emission_q

class HMM():
    def __init__(self, Initial, Transition, Emission):
        self.Initial = Initial
        self.Transition = Transition
        self.Emission = Emission
        self.N = len(Initial)
        
    def forward(self, Obs_seq_y):
        alpha = np.zeros((self.N,len(Obs_seq_y)))
        #Initialization
        alpha[:,0] = self.Initial * self.Emission[:, Obs_seq_y[0]]
        norm = []
        norm.append(np.sum(alpha[:, 0]))  
        #Induction
        for i in range(1, len(Obs_seq_y)):
            for yi in range(self.N):
                for yi_1 in range(self.N):
                    alpha[yi, i] += alpha[yi_1, i-1] * self.Transition[yi_1, yi] * self.Emission[yi, Obs_seq_y[i]]
            #Normalization
            norm.append(np.sum(alpha[:, i]))  
            alpha[:, i] = alpha[:, i] / norm[i]
        return alpha, norm

    
    def backward(self, Obs_seq_y,norm):
        beta = np.zeros((self.N, len(Obs_seq_y)))
        #Initialization
        beta[:,-1] = 1
        #Induction
        for i in range(len(Obs_seq_y)-2, -1, -1):
            for yi in range(self.N):
                for yi_1 in range(self.N):
                    beta[yi, i] += beta[yi_1, i+1] * self.Transition[yi, yi_1] * self.Emission[yi_1, Obs_seq_y[i+1]]
            #Normalization
            beta[:, i] = beta[:, i] / norm[i+1]
        return beta
    
    def Baum_Welch(self, iteration, train, test):
        interation = []
        train_log_likelihood = []
        test_log_likelihood = []

        for i in range(iteration):
            #E-step
            alpha, norm = self.forward(train)
            beta = self.backward(train, norm)
            interation.append(i)
            train_log_likelihood.append(self.log_likelihood(train))
            test_log_likelihood.append(self.log_likelihood(test))

            print("iteration",i,"--",train_log_likelihood[i])
            print("iteration",i,"--",test_log_likelihood[i])

            counts = np.zeros((self.N, self.N))
            for i in range(len(train)-1):
                for yi in range(self.N):
                    for yi_1 in range(self.N):
                        counts[yi, yi_1] += alpha[yi, i] * self.Transition[yi, yi_1] * self.Emission[yi_1, train[i+1]] * beta[yi_1, i+1] / norm[i+1]
            
            print("counts",counts)
            #M-step
            for yi in range(self.N):
                for yi_1 in range(self.N):
                    self.Transition[yi, yi_1] = counts[yi, yi_1] / np.sum(counts[yi, :])
            
            gamma = np.multiply(alpha, beta) 
            # print("gamma",gamma)
            # print("gamma shape",gamma.shape)
            for yi in range(self.N):
                for k in range(len(self.Emission[0])):
                    #print("k",k)
                    #print("gamma[yi, train == k]",gamma[yi, train == k])
                    # print("len(gamma[yi, train == k])",len(gamma[yi, train == k]))
                    # print("len(gamma[yi, :])",len(gamma[yi, :]))
                    self.Emission[yi, k] = np.sum(gamma[yi, train == k]) / np.sum(gamma[yi, :])
        
        return interation, train_log_likelihood, test_log_likelihood

    def log_likelihood(self, sequence):
        alpha, norm = self.forward(sequence)
        return np.sum(np.log(norm))/len(sequence)
            
def main():
    iteration = 1
    train = load_data("textA.txt")
    test = load_data("textB.txt")
    Initial, Transition, Emission = get_init_prob_2states() 
    hmm = HMM(Initial, Transition, Emission)
    # alpha, norm = hmm.forward(train)
    # print("alpha size",alpha.shape)
    # print("norm len",len(norm))
    # beta = hmm.backward(train,norm)
    # print(alpha)
    # print(beta)
    iterations, train_ll, test_ll = hmm.Baum_Welch(iteration, train, test)
    # print("iter",iter)
    # print("train_log_likelihood",train_log_likelihood)
    # print("test_log_likelihood",test_log_likelihood)
    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot both training and testing log-likelihoods in the same plot
    plt.plot(iterations, train_ll, label='Training Log-Likelihood')
    plt.plot(iterations, test_ll, label='Testing Log-Likelihood')

    plt.title('Training and Testing Log-Likelihood vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig('hmm_training_plot.png')

    # Display the plot
    # plt.show()

if __name__ == "__main__":
    main()

