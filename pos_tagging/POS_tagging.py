from nltk.corpus import brown
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

class Pos_tags:
    init_state = []  #
    tag = []
    words = []
    transition_matrix = [] # transition probability 
    state_likelihood = [] # state likelihood
    def initial_probability(self, corpus):    
        for i in range(len(corpus)):
            self.tag.append(corpus[i][1])
            self.words.append(corpus[i][0])
        self.tag = list(np.unique(self.tag))    
        self.words = list(np.unique(self.words))

        # calculate frequency of each tags 
        self.init_state = np.ones(len(self.tag))
        self.init_state[self.tag.index(corpus[0][1])] = self.init_state[self.tag.index(corpus[0][1])] + 1
        for i in range(1, len(corpus)):
            if(corpus[i-1][0] == '.' and corpus[i][0].isalpha()):
                self.init_state[self.tag.index(corpus[i][1])] = self.init_state[self.tag.index(corpus[i][1])] + 1  
        # calculate initial state probability
        self.init_state = self.init_state/np.sum(self.init_state)
    
    def transition_matrix(self, corpus):
        n = len(self.tag)   # the number of unique tags
        self.transition_matrix = np.ones((n, n))
        for i in range(1, len(corpus)):
            self.transition_matrix[self.tag.index(corpus[i-1][1])][self.tag.index(corpus[i][1])] = self.transition_matrix[self.tag.index(corpus[i-1][1])][self.tag.index(corpus[i][1])] + 1
        for i in range(n):
            total_sum = np.sum(self.transition_matrix[i]) 
            for j in range(n):
                self.transition_matrix[i][j] = self.transition_matrix[i][j]/total_sum
    
    def state_likelihood(self, corpus):
        n = len(self.tag)
        m = len(self.words)
        self.state_likelihood = np.ones((n, m+1))
        for i in range(0, len(corpus)):
            self.state_likelihood[self.tag.index(corpus[i][1])][self.words.index(corpus[i][0])] = self.state_likelihood[self.tag.index(corpus[i][1])][self.words.index(corpus[i][0])] + 1
        for i in range(n):
            total_sum = np.sum(self.state_likelihood[i])
            for j in range(m):
                self.state_likelihood[i][j] = self.state_likelihood[i][j]/total_sum
    
    def infer_states(self, obs):  
        init_state = np.log(self.init_state)
        transition_matrix = np.log(self.transition_matrix)
        state_likelihood = np.log(self.state_likelihood)
        
        state_obs_lik = np.array([state_likelihood[:,z] for z in obs]).T

        N = state_obs_lik.shape[0] # len of state graph
        T = state_obs_lik.shape[1] # len of observations

        viterbi = np.zeros((N, T))
        back_pointer = np.zeros((N, T), dtype=int) 
        best_path = np.zeros(T, dtype=int) 

        # initialization step
        viterbi[:,0] = init_state + state_obs_lik[:,0] 
        back_pointer[:,0] = 0 

        # recursion step
        for t in range(1, T):
            for s in range(N):
                viterbi[s, t] = np.amax(viterbi[:,t-1] + transition_matrix[:,s] + state_obs_lik[s,t])
                back_pointer[s, t] = np.argmax(viterbi[:,t-1] + transition_matrix[:,s] + state_obs_lik[s,t])

        # termination step        
        best_path_prob = np.amax(viterbi[:,-1]) 
        best_path_pointer =  np.argmax(viterbi[:,-1])

        best_path[-1] = best_path_pointer
        for t in range(T-2, -1, -1):
            best_path[t] = back_pointer[(best_path[t+1]), t+1]
        
        return list(best_path), best_path_prob
    
    # train a hmm model
    def train_hmm(self, corpus):
        self.initial_probability(corpus)
        self.transition_matrix(corpus)
        self.state_likelihood(corpus)

    # predict tags given observation:
    def predict_tags(self, observation):
        obs_index = []
        for i in observation:
            if(i[0] in self.words):
                obs_index.append(self.words.index(i[0]))
            else:
                obs_index.append(len(self.words))
        best_path, best_path_prob = self.infer_states(obs_index)
        return best_path, best_path_prob
    
    def calculate_precise(self, observation, pred_tag):
        res_tags = np.array([tag[z] for z in best_path])
        observation = np.asarray(observation)
        correct_tag = observation[:,1]
        accuracy = np.sum(correct_tag == res_tags)/len(correct_tag)
        return accuracy
    
    def confusion_matrix(self, observation, pred_tag):
        confusion_matrix = np.zeros((len(self.tag), len(self.tag)))
        for i in range(len(observation)):
            confusion_matrix[pred_tag[i]][self.tag.index(observation[i][1])] = confusion_matrix[pred_tag[i]][self.tag.index(observation[i][1])] + 1
        return confusion_matrix
    