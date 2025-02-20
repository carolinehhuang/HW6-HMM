import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    test = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], mini_hmm['prior_p'],
                            mini_hmm['transition_p'], mini_hmm['emission_p'])

    #pulling the given observation state sequence and best hidden state sequence
    test_seq = mini_input['observation_state_sequence']
    check_seq = mini_input['best_hidden_state_sequence']

    #probabilities of the transition to and from a state should add up to 1
    assert np.sum(test.transition_p[0]) == 1
    assert np.sum(test.transition_p[1]) == 1

    forward_prob = test.forward(test_seq)
    #Asserting the value of the forward probability is same as manually calculated
    assert np.allclose(forward_prob, 0.03506, atol=0.001)

    #asserting the most likely hidden state sequence associated with the sequence of observation states outputted by the class is the same as the given sequence
    assert((test.viterbi(test_seq)) == check_seq).all()


    #test edge cases where input observations are not found in the observation states
    with pytest.raises(ValueError):
        test.forward(np.array(["sunny", "rainy", "cloudy with the chance of a ValueError"]))

    # test edge case where the dimensions of inputted data is incorrect
    with pytest.raises(ValueError, match=r"Each emission must have an associated probability for each state, which also has a probability of occurring"):
        test = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], mini_hmm['prior_p'], np.array([1]), mini_hmm['emission_p'])


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    test_full = HiddenMarkovModel(full_hmm['observation_states'], full_hmm['hidden_states'], full_hmm['prior_p'],
                             full_hmm['transition_p'], full_hmm['emission_p'])

    # pulling the given observation state sequence and best hidden state sequence
    test_seq_full = full_input['observation_state_sequence']
    check_seq_full = full_input['best_hidden_state_sequence']

    forward_prob = test_full.forward(test_seq_full)

    #assert that the outputted sequence from the class is equivalent to the expected sequence
    assert ((test_full.viterbi(test_seq_full)) == check_seq_full).all()

















