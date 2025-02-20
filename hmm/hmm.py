import numpy as np
import pytest


class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        if len(self.transition_p) != len(emission_p) or emission_p.shape[1] != len(observation_states):
            raise ValueError("Each emission must have an associated probability for each state, which also has a probability of occurring")


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        #edge case for if an unexpected observation is given as a parameter
        check_observations = np.isin(input_observation_states, self.observation_states)
        if all(check_observations) is False:
            raise ValueError("There are more input observation states than observed states")

        #edge case for if there are less than 2 observed states
        if len(self.observation_states) < 2:
            raise ValueError("There must be at least 2 observation states")


        # Step 1. Initialize variables
        num_obs = len(input_observation_states)
        num_states = len(self.hidden_states)
        first_obs = self.observation_states_dict[input_observation_states[0]]
        
        emissions_mat = np.zeros((num_states, num_obs))

        #initialize first column of emissions probability matrix with the prior probabilities of the hidden states * emissions probability of the observation given the state of the first observation
        for i in range(num_states):
            emissions_mat[i, 0] = self.prior_p[i] * self.emission_p[i, first_obs]

        # Step 2. Calculate probabilities
        for i in range(1, num_obs): #for every observation other than the first, since it is used for initialization
            for j in range(num_states): #assign a probability for each hidden state at each observation state
                #all the previous forward probabilities calculated from the last observation for hidden state j
                prev_forward = emissions_mat[:, i-1]
                #get total weighted probability of observing hidden state j through all possible paths by multiplying the previous forward probabilities with the transition probabilities from each hidden state to current state j
                total_prob = np.sum(prev_forward * self.transition_p[:, j])
                #multiply the emission probabilities for state j with total weighted probabilities
                emissions_mat[j,i] = self.emission_p[j, self.observation_states_dict[input_observation_states[i]]] * total_prob

        # Step 3. Return final probability
        #return the final probability of observing the final sequence encoded in the last column, which is the sum of the probabilities of reaching the sequence through all possible paths
        total_probability = np.sum(emissions_mat[:,-1])

        return total_probability


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """
        check_observations = np.isin(decode_observation_states, self.observation_states)
        if all(check_observations) is False:
            raise ValueError("There are more observation states to decode than observed states that are given")


        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(self.hidden_states),len(decode_observation_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))

        num_obs = len(decode_observation_states)
        num_states = len(self.hidden_states)
        start_state = decode_observation_states[0]
        state_path = np.zeros((num_states, num_obs), dtype = int)


       # Step 2. Calculate Probabilities
        for i in range(num_states):
            #initialize the probabilities of hidden state at each step with the prior probabilities of hidden states * the probability of emission given the starting states of the observation states
            viterbi_table[i, 0] = self.prior_p[i] * self.emission_p[i, self.observation_states_dict[start_state]]

        for i in range(1,num_obs):
            for j in range (num_states):
                #probability of reaching hidden state j for all possible transitions from previous states
                state_prob = viterbi_table[:, i -1] * self.transition_p[:,j]
                #store the argument of the previous state that gives the max probability of observing the current observation state
                state_path[j,i] = int(np.argmax(state_prob))
                #store the highest probability of reaching this state for observation i in the sequence given the emission probability of the observation
                viterbi_table[j,i] = np.max(state_prob) * self.emission_p[j, self.observation_states_dict[decode_observation_states[i]]]

            
        # Step 3. Traceback
        best_prev_state = np.argmax(viterbi_table[:, -1]) #best previous hidden state is the state that gives the max probability of getting to the final observation
        best_path[num_obs-1] = best_prev_state #put the best previous hidden state for the final observation into the path of hidden states
        k = num_obs - 2
        while k > 0:
            best_prev_state = state_path[best_prev_state, k+1] #get the previous state that gave the max probability of getting to the observation k
            best_path[k] = int(best_prev_state) #insert the state to the front of the list
            k -= 1

        # Step 4. Return best hidden state sequence 

        return [self.hidden_states_dict[state] for state in best_path]