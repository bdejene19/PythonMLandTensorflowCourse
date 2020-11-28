'''
Have previously been using algorithms that relied on data


Hidden Markov Models instead work with probability distributions
    e.g. calculating the probability of it raining given probability of cirumstances --> e.g. simulation environment is where it could be applied

How do we know this probability?
    A lot of probabiliteis are either known or are calculated from data making these data sets really useful
    Note: we will be using predefined probability distributions rather than calculating them ourselves

Hidden Markov Models is a finite set of states: e.g. hot day or cold day
    These states are called hidden --> we never access or look at these states while interacting with the model
    Instead we have observations --> if it is hot outside, bemnet has a 80% chance of being happy, if cold, I have a 20% chance of being happy
    Each state has its different observations and the different probabilities of those observations occurring

Transition --> the probability defining the likelihood of transitioning to a different state
    For each state, we can either transition into all other states or a defined set of states


Point of hidden markov models is to be able to predict feature events from the probabilities of previous observations

Cold days are encoded by a 0 and hot days are encoded by a 1.
The first day in our sequence has an 80% chance of being cold.
A cold day has a 30% chance of being followed by a hot day.
A hot day has a 20% chance of being followed by a cold day.
On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.
'''

import tensorflow_probability as tfp
import tensorflow as tf

# first create a tfp distribution variable so you don't need to call it over and over
tfd = tfp.distributions

# categorical distributions: parametized by probabilities or log probabilities of k # of classes
# Categorical distributions closely related to OneHotCategorical and Multinomial distributions
initial_distribution = tfd.Categorical(probs=[.8, .2]) # our initial first day probability of being cold
transition_distribution = tfd.Categorical(probs=[[.7, .3], [.2, .8]]) # probability of transition i.e. cold day followed by hot/cold day

# we use the means (loc=) and standard deviations rr(scale=) to create our tensor observation distribution
# both values float tensors --> therefore, need to add decimal point at the end even if an integer
# remember tfd is a tf distribution --> we are just picking Normal distribution
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# NOTE: for the distributions above --> since we have 2 states, we have 2 probabilities, 2 means, 2 STDs ect



# create hidden markov model
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7 # steps is how many times we run through this probability model, predicting 7 days
)

# using model.mean --> calculates mean probability
# model.mean() --> is a partially defined tensor computation
# to do this, need to create a new session in tf, and run mean part of graph with numpy
mean = model.mean()

# NOTE: creating a session allows you to run part of your code without needing to run the rest
# when we call numpy on our mean variable --> gives us our temperature predictions for the next 7 days (7 steps)
'''
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
'''
# hidden markov models decrease in accuracy as num steps increases
# Hidden Markov Models are generally not that useful ==> however, can be used situationaly

