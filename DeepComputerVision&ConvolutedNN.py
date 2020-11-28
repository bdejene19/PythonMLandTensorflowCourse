'''
Deep Computer Vision --> we will be using it for classification, however, can be used for object/facial recognition and detection
    Used for tesla self driving cars, goal line technology, etc.

Topics:
    Convolution Neural Network (CNN) --> essentially the way we doo deep learning
    Image Data --> difference between it and regular data
    Convolutional layers and pooling layers --> stacks of these work together to form our convolutional base of CNN
    Work with pretrained models to perform classification tasks for us
 
'''

# Image data: previously in our NN, it was a 2D image (had a width and height)
# however, with an image, we have 3 dimensions ==> height, width, and color channels
# first layer tells us all the blue pixel values, second layer all the green, and third all the red (RGB color scheme) ==> these are our colour channels
# then we have our height and width ==> this is how images are represented in three dimensions in our computer


'''
Convolution Neural Network:
    In previous image classification (MNIST data set) --> looked at entire image at once and determined features in specific areas 
        When it learned these panterns --> it learned this in specific areas of the screen 
            e.g. a cat classification --> may classify it as cat since its eyes are on the left side of the screen
            however, may not classify it as a cat if the eyes were placed on the right side
            
    
    Therefore, a deep convolutional neural networks (CNN) looks at variables globally and learns patterns in specific areas
        i.e. cannot learn local patterns and apply  them to other areas of the image
        e.g. looking at a cat --> looking for patterns of ears, the eyes, the nose, paws, etc. 
            These features would tell us their a cat --> purpose of CNN ==> can find patterns throughout the image rather than specific location

CNN --> will scan through entire image ==> finds features in image
    Passes features through a DNN or classifier --> looks at the presences of different features that make different objects 

CNN gives us an output feature map --> tells us about the presence of features in our image (called filters in our image)

NOTE: DNN output --> a bunch of numeric values, while CNN layer is outputing a feature map ==> quantifies presence of filter patterns in different locations

We run many different filters over our image at a time, telling us about all the different places a feature is present


How a CNN works - e.g. 5x5 pixel image 
    Each CNN layer has a few properties to it: input size, # of filters, and sample siz filters
    A filter is some pattern of pixels 
    At each convoutional layer, we look at many different filters
        typically x32, sometimes 64 or 128 

Our filters are our trainable parameter in our CNN 
    The amount of filters and what they are will change as the program goes on (as we figure out what features make up a image)


Filters begin completely random and change over time:
    e.g. imagine having an image of an X 5x5 pixels ==> lets say we 3x3 pixel filters (32 different combinations)
        Therefore, CNN will look at 3x3 sections of our image and try and find how well the match filters
            They will output a feature map (which is a little bit smaller than origianl image) --> presence of different features in diff areas
    Note: if you have 2 filters in your CNN --> it will output feature maps of depth 2 (a feature map for each filter)  

Our 3x3 image section then does the dot product with the filter --> giving us a value for each pixel of the 3x3
    Value of dot product will tell us how similar the pixel of our image is to our filter

After creating feature map of 3x3 filter/image section, to use second filter, image section moves over a column (i.e. get some overlap  
    Then move down a row and do the same thing

However, we're going to do this for 32, 64 or even 128 filters --> therefore, will have a ton of layers
    So we will be continuously expanding the depth of the output feature as we move through the convolutional layer
        Therefore, this operation can be quite slow --> this is why we can talk about pooling   

e.g. in our image example, we are originally comparing individual pixels --> determinge if their black or white (grey scale)
    But then we're trying to determine what a line is, then maybe an edge, and more complex features 

Layers of a CNN stacking on top of each other --> it is easier find moe complicated shapes


Sometimes we want our outpu feature map to be the same dimensions as our image 
    To do this, we need to add padding to our image ==> adding blank pixels 
        Therefore, now we could take a 3x3 filter where each pixel is centered 
            Allows us to generate an output map the same size as input, and look at features on the edges of images that we wouldn't otherwise be able to see


Strides: strides is how many pixels you move your filter when creating your feature map 
    Note: will use a different stride in different instances 

Pooling: we have a lot of layers, there needs to be a way to make it simpler
    Have 3 main types of pooling operations: min, max, and avg ==> take specific values from output feature map 
    Max pool - tells maximum presence of feature in a local area (only realy care if a feature exists or not)
    Avg pool - not used that often ==> tels if you avg presence
    Min pool - tells if you if it does not exist 
    
    To reduce featue map dimensionality --> sample 2x2 (typically) of our output feature map and take the min, max, or avg of those 4 values 
        Maps the feature map to a new output feature map that is half the size
'''


# willing be using keras and CIFAR image dataset from tf to make 1st CNN

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# first need to load dataset
# Note: it is being loaded as a weird tf object, not the same load up method as before

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# want to normalize data
(train_images, test_images) = train_images / 255.0, test_images / 255.0

# different classification types that our image could be
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


plt.imshow(train_images[1])
# train labels takes multiple indexs since there are multiple lists in it ==> still get a single number (9) to index in class_names
plt.xlabel(class_names[train_labels[1][0]])
#plt.show()


# CNN Architecture
# Essentially stack Convolutional and max/min/avg (some kind of pooling layer but dont have to)
# first layer (convolutional): defines the amount of filters, sample size (how big the filters are), activation function, and the input shape
# input shape is what we expect in first layer ==>don't need to do it for future layers since it will figure it out by itslef
# second layer (pooling layer) ==> takes in a tuple parameter (e.g. 2, 2) ==> have pooling dimensions of 2x2 and stride of 2
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3))) # Note: we have more filters in our 2nd and 3rd convolutional layer
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3)))

# can get a summary using tf summary
# NOTE: we did the sampling without padding
# therefore, for each convolutional layer we have 2 less pixels ==> instead of 32, we have 30
# pooling layer basically just shrinks it ==> can specify by how much, but by 2 in our case
model.summary()

# however, our summary doesn't really tell us muc about the data ==> only tells us of presence of certain features as we go through convolution base
# therefore, now we need to pass this information into a dense layer classifer
# dense layer classifier --> takes pixel data of specific features from convolution base and maps it to one of the class names

# therefore, convolution base extracts features while dense layer classifies them
# adding dense layer is fairly easy, however, first need to flatten our feature data (from (4, 4, 64)) to a line
model.add(layers.Flatten()) # flattened value is equivalent to multiplying dimensions values (therefore now equals 1024)

# then add dense layer to feed convolutional data into ==> has 64 neurons with relu activation function
model.add(layers.Dense(64, activation='relu'))

# add output layer with 10 neurons since we have 10 classes
model.add(layers.Dense(10))

# now when we re-run summary method, can see output of additional dense layers
model.summary()


# now need to train the data
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#trained_model = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
# had 76% accuracy ==>  did not evaluate  since did not want to go through run time again



# working with small data sets (few thousand data) makes it hard to pick up features
# however, you can using some techniques


# 1) Data Augmentation --> take one image and turn it into several different images that you can pass through your model
    # e.g. stretch, compress, rotate, etc of the same images ==> helps a lot when you don't have different unique images
    # is still better to have unique images, but data augmentation allows you to pick up features that may be in a different orientation, streched or compressed

# data augmentation --> will be using keras image ImageGenerator to do so
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

augmented_data = ImageDataGenerator(
    rotation_range=40, # is the rotation degree of each image (i.e. how many orientations we look at it at)

    # width and height shift range --> are values between 0-1
    # width and height represent a floating point that you can move your image horizontally (width) or vertically (height)
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2, #shear transforms data data to put it on a slant (fix one axis while stretching the other
    zoom_range=0.2, # zoom < 1 magnifies image, zoom > 1 zooms out of the image
    horizontal_flip=True, # flips images horizontally  down on a random basis (Note: vertical flip is what turns images upside down)
    fill_mode='nearest' # 4 types of fill mode (all to different ways to fill empty data)
                          # 1) nearest (default option) - closest pixel value is chosen and repeated for all empty values
                          # 2) reflect - fills in empty space with reverse order of known values
                          # 3) wrap - fills empty space with known values in unchanged order (i.e. image begins to repeat to fill out space)
                          # 4) constant - fills points lying out side of boundary using a constant value, specify after using cval= parameter

)


# what if you've used data augmentation but you still don't have enough data
# we can used pretrained models --> companies like google make their own amazing CNN that are open source that we can use
# we will use part of a CNN they trained on 1.4 million images
# Therefore, we'll have a really good base and all we need to do is fine tune the last few layers
# beginning layers of this model is what picks up on edges and general things that appear in most images (generally)
# we can then just change the layers of the top model a bit to the problem we want
# is an effective way of using a pretrained model ==> i.e. use pretrained model as base, then fine tune to our specifications
# will still pass training data through


#  will be classifying cats vs dogs from tf data sets ==> need to import os ontop of usual imports (matplotlib, tf, numpy)
import os
import tensorflow_datasets as tfds
keras = tf.keras


# will almost always have to look at documentation at how to load  your data
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)


# lets take a look at some of the images
# this function is unique to this situation
get_label_name = metadata.features['label'].int2str # creates function to get labels of images
for image,label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))


# images are not the same size ==> therefore, we need to scale these images to be resized to same shape
img_size = 160  # better to compress images than make them bigger ==> but images will be resized to 160x160


def format_image(image, label):
    image = tf.cast(image, tf.float32) # cast converts every pixel to a float 32
    image = (image/127.5) - 1
    image = tf.image.resize(image, (img_size, img_size))
    return image, label

# reshaping all our photos for training, validation and testing
train = raw_train.map(format_image)
validation = raw_validation.map(format_image)
testing = raw_test.map(format_image)

for image,label in raw_train.take(2):
    print(image.shape)



# now need to shuffle and create batches of images to put into our model
# READ MORE ON HOW BATCHES WORK
# images are 160x160 with 3 image channels (RGB
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1000


train_batch = train.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
validation_batch = validation.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
test_batch = testing.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)



# picking pretrained model for base --> is one of the harder steps
# will be using MobileNet V2 --> from google built into tensor flow

image_shape = (img_size, img_size, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                               include_top=False, # very important --> asking if we are using the classifier that comes with it or its own
                                                weights='imagenet' # loading in weights of MobileNetV2
                                            )


base_model.summary() # this base model summary shows googles CNN base
# take features that google base determines are 5, 5, 1280 (5x5 images)
# take 5x5 feature and pass it through more of our own convolutional layers and classifiers to do our own classification (cats vs dogs)
# base model returns a tensor output of (32, 5, 5, 1280) ==> have 32 filters of 5x5 pixels of feature extraction from our original (1, 160, 160, 3)

# we want to freeze the base --> if we put in base model directly rn, it will start retraining weights and baises
# however, we don't want this ==> they have already been defined, don't wanna to manipulate it more
# to do so, we turn trainable to false
base_model.trainable = False

# take (5, 5, 1280)  output of convolutional base layer
# want to use this output to either classify cat or dog
# therefore, we use global average pooling layer ==> essentially takes average of our 1280 layers that are 5x5 into a 1D tensor
global_avg_pool_layer = tf.keras.layers.GlobalAveragePooling2D()

# add prediction layer
prediction_layer = keras.layers.Dense(1) # only need 1 neuron in prediction layer since it is either cat OR dog (one or the other)

# now need to create full model with CNN base layers from google with classifier layers that we created
model_all_layers = models.Sequential([
    base_model,
    global_avg_pool_layer,
    prediction_layer
])

# notice, we have 2.25 million parameters, but only 1281 are trainable
# this is because we have 1281 connections coming fom our previous layers (1280 weights and 1 bias)
# wee just added our global_average_pooling and dense layer ==> benefits of convolutional base
model_all_layers.summary()


# training model
# learning rate --> determines how much you are allowed to modify the weights and biases of the network
base_learning_rate = 0.0001 # low since we don't want to change our weights that much since we have base already

# compile model with optimizer and loss functions
# lr= is parameter for learning rate
# we use binary cross entropy as our loss function since we only have 2 classes (cat or dog) outputs
# if we had multiple potential outputs, would use some type of crossentropy
model_all_layers.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                         metrics=['accuracy'])

# we will evaluate the model before even training it by using the validation batches
# want to see how the model will perform with the current base model being as it is and not changing any weights

initial_epochs = 3
validation_steps = 20

# pretraining,  there is a loss of 71% and accuracy of 56% (essentially guessing cat or dog pre-training)
loss_pretrain, accuracy_pretrain = model_all_layers.evaluate(validation_batch, steps=validation_steps)


# now we train our model ==> training takes an hour (looks at lots of images)
# get 93% accuracy ==> this is pretty good considering we just used a base layer (classified 1000 images --> i.e. google CNN base layer)
# to specify it to cats and dogs, we just added a dense classifier ontop (i.e. our global avg to flatten, and dense layer)
history_of_trials = model_all_layers.fit(train_batch,
                                         epochs=initial_epochs,
                                         validation_data=validation_batch)

# since it takes an hour to train ==> don't want to wait an hour every time
# therefore, we can save the model into a keras h5 file
# type name of model, then use save function with name you want to give h5 file
model_all_layers.save('cats_vs_dogs CNN.h5')

# can load it by the following
new_model = tf.keras.models.load_model('cats_vs_dogs CNN.h5')


'''
Tensorflow for object detection 
    Tf has an object detection api ==> no time to go through it in youtube video, but should do in my own time 
'''
