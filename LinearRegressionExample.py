# step 1: imports

from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import urllib
from IPython.display import clear_output # used just to clear display

import sklearn as sk
import numpy as np # for optimized multidimensional calculation of arrays (cross product, dot product, etc)
import pandas as pd # data analytics tool, allowing us to easily manage data and look at it
import matplotlib.pyplot as plt # is a graphical tool
import tensorflow as tf


# step 2: load the training and test files
# in this project, will use linear regression on titanic data set
# first open data set, and store it in a train and eval form
df_train = pd.read_csv("train.csv")
df_eval = pd.read_csv("eval.csv")

# however, we are aiming to predict the survived value
# 0 = died, 1 = survived
#  we will pop the survived column

y_train = df_train.pop('survived') # answers to training data (since we are trying to predict survivability)
y_eval = df_eval.pop('survived') # answers to test data
# print(df_train.head()) # have survived column popped

# using .describe() function on df gives us our mean, std, and quartile distribution
# note: dataframes also have shapes. can be found using.shape
print(df_train.shape) # get (627, 9) meaning 627 entries with 9 attributes (columns)


# can make graphs of df data
hist = df_train['age'].hist(bins=20)
hist.set_title('My histogram') # this is how you label axis in matplotlib
hist.set_xlabel('Ages') # note: there are no predictive text for it, need to type it out
hist.set_ylabel('Frequency')
plt.show()

# histogram shows that most people are in 20-30, which would cause a bias in our LR algorithm
# now lets look at the sex of people
# the value_counts counts the values in the series specified, where the index is the title and how often the values occurred
# series is ordered in descending manner
# note: when choosing the kind of plot you want, kind='bar' gives vertical bar graph, and 'barh' is horizontal bar graph
hist2 = df_train['sex'].value_counts().plot(kind='barh') # creates bar graph for frequency of each type (i.e. male or female)

# data shows there were about double the amount of men than women on the titanic
plt.show()

# now we want to create a new column (axis=1) with % survival looking at our df_train and y_train
# figure out this line: pd.concat([df_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

'''
have so far learned a few things about the data:
    1) majority of passengers were in their 20's to 30's
    2) majority of passengers were male
    3) majority of passengers were third class
    4) females have a significantly higher chance of survival
'''

# step 3: hard code categorical and numerical columns of df
# categorical data is fairly common --> e.g. first, second and third class, e.g.2 male or female
# need to transform categorical data into integer values --> e.g. first (0), second (1) and third (2) class
CATEGORICAL_COLUMNS = ['sex', 'class', 'n_siblings_spouses', 'parch', 'deck', 'embark_town', 'alone']

NUMERICAL_COLUMNS = ['age', 'fare']

# step 4: loop through categorical and numerical columns to create feature column
# feature column: contains creates list of all unique  elements for each  categorical/num feature
# feature columns will allow us to create a linear list
# need to loop throw categorical data to create feature columns
feature_columns = []
for cat_feature_name in CATEGORICAL_COLUMNS:
    feat_name_vocabulary = df_train[cat_feature_name].unique() # creates a list of all unique values of the feature name

    # to add to our featue column list you append --> however, need to call tensor
    # append(tf.(name of feature column).categorical_column_with_vocabulary_list(categorical column variable, vocabulay variable))
    # creates dictionary with category name as key and category feature vocabulary as values
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(cat_feature_name, feat_name_vocabulary))

# when looping through numerical columns --> dont need to create separate variable to hold unique values
# call append function to feature_columns and use tf.feature_column.numeric_column(num feature column var name, and its dtype calling tensor (i.e. dtype=tf.float32)
for num_feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(num_feature_name, dtype=tf.float32))

print(feature_columns)

# step 5: training process
# give training data to our algorithm --> however, what if you had 25 terabytes of data?
# upload specific set of elements as you go in batches
# epochs: the amount of times the model sees the same data
# therefore, can feed the same data in a different order or form and see the same type of data to pick up patterns
# it is also possible to pass in too much data ==> i.e. memorizes answers rather classify for identification
# to override this, use smaller number of epochs and work your way up

# need to create a function that inputs your training data in batches
# turns data into tf.data.Dataset object and returns an object to use
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
        ds = ds.shuffle(1000)  # randomize order of data (for training purposes)
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(df_train, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(df_eval, y_eval, num_epochs=1, shuffle=False) # we only need 1 epoch since we are testing it, and need no shuffling

# now we need to pass in our feature columns to through a linear estimator classifier
# set feature_columns attribute to equal our feature column variable
linear_estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# linear_estimator is now our LR regression estimator
# therefore, now need to train it with sample data using built in tensorflow train method
# we are now going to train our model that has taken batches of inputs to become a dataset object
linear_estimator.train(train_input_fn) # trains our linear estimator
result = linear_estimator.evaluate(eval_input_fn) # tests with evaluate method
train_input_fn
print(result['accuracy']) # got 75% accuracy
print(result) # prints all linear estimator values


# tensorflow --> good at making a lot of predictions from lots of data
# is bad at making one prediction from few instances
# we can create another results variable, however, this time we are going to make a prediction on our eval data
# need to pass in our eval input data
result = list(linear_estimator.predict(eval_input_fn)) # say list just to put all attributes of linear estimator in a list so we can iterate or index
print(result)

# this gives us a dictionary of prediction characteristics for each evaluation data object
# however, we are only interested in the probabilities
print(result[0]['probabilities']) # this looks at data object 0 and the the probability predictions of survival

# result tells us 6% of survival, 94% of death on titanic
# we can check probability of survival by doing a third tier index of [1] since 1 = survived and 0 = dead
print(result[0]['probabilities'][1])


# can now look at a specific individual and determine if their survival odds make sense
# have a 35 year old male --> predicted survival chance: 4%
print(df_eval.loc[0])
print(result[0]['probabilities'][1])


# have a 58 year old female --> predicted survival chance: 45% ==> therefore, predictions make sense
print(df_eval.loc[2])
print(result[2]['probabilities'][1])


