

# Neural Nets can have many layers --> example 3 layers: have an input and ouput layer, however, we also have a hidden layer inbetween
'''
e.g. if you had a 28x28 pixel image you were tyring to classify --> how many input neurons would you need in your NN?
    If you are looking at entire image to make prediction, you will need all pixels (28*28 = 784)
    Therefore, would have 784 input neurons --> however, if we only had one pixel, would only need one input neurons


Output layer --> contains output neurons (as output pieces we want) ==> many ways to represent output layer
    e.g. if we had 2 classes we wanted to represent, its output neuron will give a number between 0 and 1 inclusive
        If the value is closer to 0, we can call it class 0, if its closer to 1, it would be class 1

    Therefore, when we have our training data, the output would either have to be 0 or 1. This works sometimes

Typically in classification --> it makes most sense to have the same number of output neurons as classes
    e.g. imagine having 3 input neurons and 5 output neurons --> each output would have a value between 0 and 1
        sum of ouput neurons would equal 1 --> therefore, looks like a probability distribution

Essentially, we are making predictions for how much our input information is for each one of our classes
    If we were doing a regression task, it would just have one neuron


In NNs --> can have many different types of hidden layers --> can connect to other hidden layers if they wanted
    But how are the layers connected (input, hidden and output)? They are connected through something called weights
        Can have different architecture of connections
        When all inputs connect each hidden neuron --> is called a densely connected NN

Weights --> is what the NN changes and optimizes to determine the mapping of input to output
    Each weight is typically between 0 and 1, but can also be large or even negative (depends what kind of network your doing and how its designed)

Weights are considered the trainable parameters of our neural network --> is what gets tweaked to make predictions

We also have something called biases ==> are different to regular nodes
    There is only one bias --> and bias exists in the previous layer that it affects ==> connects each neuron so still DCN

Bias does not take in any input information --> it is another trainable parameter for the network
    Bias is just some constant numeric value --> the bias weight is typically 1

Note: a bias from one layer does not have a weight to a bias in the next layer


Lets say we are trying to predict a colour --> red (0) or blue(1)
    Takes in our 3 inputs x, y, z ==> what we ned to do is figure out how to find values of hidden layer nodes
    The way to determine these values is equal to the weighted sum of all the previous nodes connected to it

Initially at the beginning --> the weights for our input-hidden layer are random
    As NN gets better, will update and be changed to make more sense


Training Process and Activation function
    Activation function --> we want our NN to ouput a value between 0 and 1
        however, with random weights at the beginning --> e.g. how could we use a value of 700 to classify it as red or blue?
        This is where we use an activation value

    Activation fn 1) Relu (rectifeid linear unit) ==> takes any values negative values and makes them 0
                  2) TanH (HyperBolic Tangent) ==> Squishes values between -1 and 1
                  3) Sigmoid ==> squishes values between 0 and 1


    Why use an activation function on an intermediary hidden layer?
        To introduce complexity into our NN --> if we can introduce a complex activation function intothis process, can make more complex predictions
            A NN without an activation function is essentially just a linear regression model
        Activation function allows you to make more complex predictions by moving data into different dimensions


How Neural Networks Train:
    Weights and baises are what your network comes up with to make better predictions.
    But now we are going to talk about a loss function
        Previously, we would just compare our predicted output to the actual output to judge our accuracy

    Say we make a prediction in our NN using the sigmoid activation function --> we may get a value of 0.7 still, but far away from 0 if output was red
        But how far away? This is when we use the loss function --> calculates how far away our output was from our expected output
            i.e. tells us how good or bad our network was --> high loss means we need to tweak weights and biases to move network in different direction

        common loss/cost (synonomous) functions:
            1) Mean Squared Error -->       2) Mean Absolute Error         3) Hinge Loss



But how do we update these weights and biases?
    This is what we call Gradient Descent
    Parameters of our network are weights and biases - by changing them, our network will either get better or worse

Looking at the gradient descent map --> we're essentially trying to optimize the loss function
    To do that, we are looking for a global minimumum ==> this is the process of gradient descent
        i.e. moving your loss function to its most minimal state


NN RECAP:
    Have inputs and outputs with hidden layers (sometimes multiple)
    Use activation functions to shift our data to make data to make better predictions for complex problems with bias
    Information is passed through layers by weighted sum of all the connected neurons to it, add bias weight constant, then put it through activation function
    Then make our way to output layer
    When training --> we make predictions, then calculate how wrong our predictions were using loss function
    Use loss function to calculate our gradient descent (where advanced math) --> where we need to move to minimize loss
    Then use an algorithm called back propogation to step back through the network and update the weights and biases according to the gradient calculated


Unless you overfit, the more data you see, or the more epochs you have, your network will be getting better and better at predictions

Optimizers: what allows you to reverse propagate through neural network
    There are many different types - e.g. Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent, Momentum, etc.
    NOTE: will not be covering optimizers, is more of an advanced ML technique
'''

import tensorflow as tf
from tensorflow import keras

# helper lib rary
import numpy as np
import matplotlib.pyplot as plt

# for this tutorial --> using built in keras MNIST Fashion dataset (included in keras)
# time to load the data_set
mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data() # tuples split data into training and testing sets

# mnist data set --> essential pixels of data for collection of fashion clothing
# lets look at shape --> see we have 60000 data entries, with 28 lists, with 28 items inside
print(training_images.shape)

# to index our training images --> Note: is type is np.ndarray
# indexing work similar to pandas ==> however, takes in 3 parameters
# one pixel is represented between a number between 0 and 255 ==> in this case works with grey scale
# 0 = black, 255 = white
# however, you could have numbers between 1-255 for every pixel for RGB colours
print(training_images[0, 23, 23])

print(training_labels[:10]) # grey scale for first 10 pixels

# Label names --> will hold an integer value between 0-9 since there are 10 different types of articles of clothing
class_names = ["T-shirt", "Trousers", "Overalls", "Dress", "Coat", 'Sneaker', 'Shirt', 'Sandal', 'Bag', 'Ankle boot']
# Data preprocessing --> last step befoe creating model
# We are going to apply some prior transformations to our data before feeding it the model
# In this case, we are making sure each of the greyscal pixel values is between 0 and 1
# Do this by dividing each pixel value by 255
# Note: need to do this to both training and test data ==> want our data to be fed in the same way to predict outputs
training_images = training_images / 255.0
test_images = test_images / 255.0


# preprocess data since we want our values to be as small as possible since we initially set random weights and biases
# therefore, if we have massive data and small weights and biases --> system will have to work harder to reweight


# time to create model --> pretty easy
# Note: data is the hardest part of ML and NN --> getting data in the right shape, form, and preprocessed correctly
# doing ML algorithm for model is fairly easy because tools like tensorflow are used
# In our model - we use keras.layers.Sequential --> the simplest form of a NN ==> passes through layers one at a time
model = keras.Sequential([
    # Flatten method: takes 28x28 pixel and flattens it into a line basically
    keras.layers.Flatten(input_shape=(28, 28)), # first layer: input layer

    # Dense layer: means every neuron of this current layer is connected to the previous layer
    # 128 resembles number of neurons we want in our second layer --> typically better to have less neurons in 2nd layer
    # making use of activation function setting it to relu --> rectified linear unit (gets rid of - numbers)
    keras.layers.Dense(128, activation='relu'), # second layer

    # 10 neurons in our 3rd output layer e we have
    # Note: we have 10 output neurons since output layer should have as many neurons as classes
    # soft max activation: makes sure all values of our neurons add up to 1 and are between 0 and 1 (prob. distribution)
    keras.layers.Dense(10, activation='softmax')
])

# model has defined the basic architecture of our NN
# Have decided amount of neurons in each layer, the type of layer, and the activation function used


# now need to look at optimizer and loss functions we want to implement
# Note: we will be using 'adam' and 'sparse_categorical_crossentropy'
# READ ABOUT OPTIMIZERS AND LOSS FUNCTIONS IN OWN TIME
# metrics is the output we're looking for --> accuracy of our model
# if we wanted, we could pick different values via hyper parameter tuning
    # i.e. we can't change the weights and biases, however, we have control over what parameters we're interest in
    # e.g. # number of neurons in a layer, optimizer, loss function ==> are all hyperparameters
        # look at how they affect and change your model

# use compile function to put together model with optimizers and loss functions, and your metric of interest
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# time to train data with .fit (train word for NN)
# note: number of epochs is another hyper parameter
model.fit(training_images, training_labels, epochs=10)

# training the data takes more time since there are images --> but the NN prints the loss and accuracy for each epoch (time it sees the data (i.e. 5th time seeing same data vs 6th time))
# model tells us that we have 91% accuracy from our training ==> however, to get true accuracy of model, need to test NN
# verbose --> is essentially answering if we are looking at output or not ==> how much is being printed to the console
(test_loss_value, test_accuracy_value) = model.evaluate(test_images, test_labels, verbose=1)

# Our training data --> suggested 91% accuracy, but when tested with new images, was only 88% accurate
# this is an example of over fitting --> 10 epochs allowed the neural net to memorize some answers to increase accuracy
print(f'Test neural net loss value: {test_loss_value}\nTest neural net accuracy value: {test_accuracy_value} ')


# have trained and tested the data --> but time to make predictions
# Note: when using the predict method on your model, need to pass in an array since that is the expected data type
predictions = model.predict([test_images[0:100]])

# have array within an array containing predictions of first 100 elements ==> however, the numbers are so exponentially small
# how do we actually tell which class_name label it should be given?
# solved using convenient np.argmax

# returns the index of the highest value in the list
print(np.argmax(predictions[0]))

# algorithm returns index 9 --> therefore, looking at class names==> we can predict it would be an Ankle Boot
# can later add other components where it asks for user input, makes a prediction on the data set with the probability, then shows the actual image on the screen for the user
