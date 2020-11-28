# Rather than predicting a numeric value (% survival chance) in LR;
# instead will be looking at a specific data entry is classified in the proper class
# will be using the iris data set

import tensorflow as tf
import pandas as pd

# import and read the iris data training and testing data sets
# fetch data training and testing data set from server using tf keras
train = tf.keras.utils.get_file("iris_training.csv",
                                     origin="http://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")

eval = tf.keras.utils.get_file("iris_test.csv", origin="http://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# flower species is what we want to try and predict list
FLOWER_SPECIES = ['Sertosa', 'Versicolour', 'Virginica']

# creating names of df columns in
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# making a df using custom column names created
# note: df_trainer holds the file of the csv, and now the df is being created
trainer = pd.read_csv(train, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(eval, names=CSV_COLUMN_NAMES, header=0)


# pop column trying to predict to hold answers to training and testing data
y_train = trainer.pop('Species')
y_eval = test.pop('Species')

# need input function to enter data into our classifier --> was copy and pasted
# Note: this input function works differently than the one for linear regression
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


# now make feature columns
# Note: since all our columns are numeric, don't need a separate loop for categorical columns
feature_columns = []
for key in trainer.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

# time to build classifier model
# tf has tons of built in classifier --> premade model that we feed information
# for this type of classification problem, have 2 built in tf classifier options:
#       1) DNNClassifier (Deep Neural Net)        2) LinearClassifier --> works similar to LR
#
# in linear classification, instead of predicting numeric value like LR, instead are predicting a classification (label)


# for this classification problem, will use DNNClassifier model (since tf web said its better)
# building a neural net with 2 hidden layers with 30 nodes in the first layer, and 10 in the second
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, # pass feature columns just like linear classifier

    # 30 nodes in first layer, 10 in second layer
    hidden_units=[30, 10],

    n_classes=3 # n_classes we decide ==> 3 since we have 3 types of flower
)

# training DNNClassifer model
# this is different than training our LR approach ==> need to call lambda
# lambda --> beneficial since it allows you to define functions in one line of code
# NOTE: input_fn= is a parameter, not your input_fn ==> input_fn parameter allows us use lambda to call our input_fn function
classifier.train(
    input_fn=lambda: input_fn(trainer, y_train, training=True),

    # steps are like epochs, but instead of saying we want to see the data 5000 times, it is saying it will stop after seeing 5000 data points
    steps=5000
)


# time to evaluate our classifier ==> works similarily to training
results = classifier.evaluate(
    input_fn=lambda: input_fn(test, y_eval, training=False)
)

# accuracy of 55% ==> not the best
# print(results['accuracy'])



# now if we wanted to predict the flower type from user input:
# need to build a second input_fn though


def input_fn2(features, batch_size=256):
    # converts inputs to a data set withoutlabels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predictions = {}

for feature in features:
    var = (input(f'{feature}: '))

    if not var.isdigit():
        print('value not specified')

    else:
        predictions[feature] = [float(var)] # NOTE: float requires to be in a list to run


predict = classifier.predict(input_fn=lambda: input_fn2(predictions))

for pred_dict in predict:
    print(pred_dict)
    class_id = pred_dict['class_ids'][0] # gets our class id --> recall: our flowers can be number 0, 1, 2
    classifier_prediction = pred_dict['probabilities']
    print(f'Probability of flowers: {classifier_prediction}')
    print(f'Most likely {FLOWER_SPECIES[class_id]} \nProbability: {classifier_prediction[class_id]}')

