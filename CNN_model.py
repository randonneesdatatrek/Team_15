from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Get the data
tr_data = ImageDataGenerator()
train_data = tr_data.flow_from_directory(directory="/Train", target_size=(224,224))
ts_data = ImageDataGenerator()
val_data = ts_data.flow_from_directory(directory="./Validation", target_size=(224,224))

# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))
# 1st conv block
model.add(Conv2D(64, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# 2nd conv block
model.add(Conv2D(32, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
# 3rd conv block
model.add(Conv2D(16, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# fit on data for 30 epochs
model.fit_generator(train_data, epochs=30, validation_data=val_data)
