####### Import libraraies
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.layers import Conv2D
import glob
import keras
import numpy as np
from keras import models
from keras import layers
######


###### model
classes = ["1", "2", "3"]
nbClasses = 3

# ___________________ Training _________________________________________
path_to_imagesTrain = "train/"
LabelTrain = 0
x_train = np.empty((42, 224, 224, 3))  # 80% of data for train and val
y_train = []
for cl in classes:
    paths_list = glob.glob(path_to_imagesTrain + cl + "/*")
    y_train += [LabelTrain] * len(paths_list)
    for i, image_path in enumerate(paths_list):
        image = keras.preprocessing.image.load_img(image_path)
        image = keras.preprocessing.image.img_to_array(image)
        image = image[:, :, :3]  # alpha dernier
        image = preprocess_input(image)
        x_train[i] = image
    LabelTrain += 1
y_train = keras.utils.to_categorical(y_train, nbClasses)

# ___________________ Test ___________________________________________
path_to_imagesTest = "Test/"
LabelTest = 0
x_test = np.empty((11, 224, 224, 3))  # 20% of data for test
y_test = []
for clTest in classes:
    paths_listTest = glob.glob(path_to_imagesTest + clTest + "/*")
    y_test += [LabelTest] * len(paths_listTest)
    for iTest, image_pathTest in enumerate(paths_listTest):
        imageTest = keras.preprocessing.image.load_img(image_pathTest)
        imageTest = keras.preprocessing.image.img_to_array(imageTest)
        imageTest = imageTest[:, :, :3]  # alpha dernier
        imageTest = preprocess_input(imageTest)
        x_test[iTest] = imageTest
    LabelTest += 1
y_test = keras.utils.to_categorical(y_test, nbClasses)

# ___________________ Model ________________________________________
def build_model(input_layer, start_neurons):
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv5)
    conv5 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv5)
    conv5 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv5)
    pool5 = MaxPooling2D((2, 2))(conv5)
    pool5 = Dropout(0.5)(pool5)

    # Middle
    convm = Flatten()(pool5)
    convm = Dense(128, activation='relu')(convm)
    convm = Dense(nbClasses, activation='softmax')(convm)

    deconv5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Dropout(0.5)(uconv5)
    uconv5 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv5)
    uconv5 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv5)
    uconv5 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv5)
    uconv5 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv5)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)
    return output_layer

input_layer = keras.Input(shape=(32,))
output_layer = build_model(input_layer, 16)
