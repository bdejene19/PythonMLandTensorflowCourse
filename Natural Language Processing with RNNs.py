'''
RRN = recurrent neural networks

Natural Language Processing (NLP) is a displine of computing
    Deals with understanding natural human languages ==> is done with RNN

examples of NLP --> autocompletion, auto-correct, voice-assistance, translations


Recurrent Neural Networks --> are very good at classifying and understanding text data ==> however, fairly complex

We're gonna focus on why RNNs work the way the work, and when you use RNNs

First thing we're going to do is a sentiment analysis, then do a character generation (to generate the next character in a sequence of text)
    We'll generate enough text to write a play ==> it will read through romeo and juliet
        will then give it a prompt --> will then write the rest of the play



How do we turn textual data into numeric data that we can feed into our NN?
    Bag of words - is a famous algorithm to convert textual data to numeric ==> is flawed since it only works for simple text
        Will look at entire training data set ==> then create a dictionary look up of the vocabulary
        Every single unique word in our data set is the vocabulary  will be placed in the dictionary (our word index table)
        Have a word represents a unique integers

        Called bag of words since we're gonna look at a sentence and keep track of the words that are present and the frequency


    Bag of words --> have a number for each unique word ==> result is 2 dictionaries
        dict1) key of unique words to integer representation ('this': 0)
        dict 2) integer reprentation of frequency of word (0 : 2) ==> means the word this is seen twice in bag



The way a RNN works --> takes in one word at a time as sequence, rather all at once
    After one word, you get a numeric output after analysis, and spits out an output
    You then move onto the second word, but then it outputs a value considering the first and second word ==> then so on and so forth
        Network uses what its seen previously to understand what it sees in relation to what came befor

This is considered a simple RNN network layer - works fairly well ==> however, issue is if we have very long sequences
    The beginning of sequences get lost as we go through it
        As you have more words, gets increasingly difficult to understand thins in relation to the beginning, since it is not much less significant


Next layer - LSTM ==> Long-Short Term Memory ==> adds another component that keeps track of internal state
    Previously, all we were keeping track of internal state was previous output ==> e.g. at time 1 (word 2), we have the output of time 0 (word 1)
        In LTSM --> we are adding the ability to access any previous output from any previous state we've looked at
        This allows us to add some complexity to the model (LTSM almost acts as a time line)
    LTSM allows us to keep track of things at the beginning and middle for big pieces of text ==> since we have ability to access previous outputs

'''

# example on sentiment analysis on movie reviews
import tensorflow as tf
from keras.preprocessing import sequence
from keras.datasets import imdb
import os
import numpy as np


VOCAB_SIZE = 88584 # num of unique words from movie reviews

(train_data, train_label), (test_data, test_label) = tf.keras.imdb

