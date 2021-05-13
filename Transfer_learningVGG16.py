###### Import libraries
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import glob
import keras
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
#######


###### model
def model_VGG():
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
    
    print("My model")
    base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # on cree le modele VGG19
    base.summary()
    for i, layer in enumerate(base.layers):
        print(i, layer.name)
    for layer in base.layers[:11]:
        layer.trainable = False
    for layer in base.layers[11:]:
        layer.trainable = True
    modelCNN = models.Sequential([
        base, 
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # fully-connected layer
        layers.Dense(nbClasses, activation='softmax')  # softmax layer
    ])
    modelCNN.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    modelCNN.summary()
    modelCNN.fit(x_train, y_train, epochs=2, validation_split=0.2)
    result = modelCNN.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', result[0])
    print('Test accuracy:', result[1])
    
    # ___________________ Performance metrics_______________________________________
    test_generator = ImageDataGenerator()
    test_data_generator = test_generator.flow_from_directory(path_to_imagesTest, target_size=(224, 224), batch_size=32, shuffle=False)
    test_steps_per_Epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
    predictions = modelCNN.predict_generator(test_data_generator, steps=test_steps_per_Epoch)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())
    CM = confusion_matrix(true_classes, predicted_classes, normalize='all')
    print('Confusion matrix : ')
    print(CM)
#####

model_VGG()
