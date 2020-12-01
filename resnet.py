from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.datasets import cifar10
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
import numpy as np


#----------Data Loading and Preprocessing----------#
# function for one hot encoding
def convert_to_one_hot(Y, C):
    
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# load cifar10 data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# convert labels to one hot encodings
Y_train = convert_to_one_hot(Y_train, 10).T 
Y_test = convert_to_one_hot(Y_test, 10).T

# preprocess train and test data
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
#--------------------------------------------------#



#------------------Make the model------------------#
# base pre-trained model, without the dense layers
base_model = ResNet50(input_shape=(32, 32, 3), weights="imagenet", include_top=False)

# add global average pooling layers and dense layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.6)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.6)(x)
preds = Dense(10, activation="softmax", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

# final model
model = Model(inputs = base_model.input, outputs=preds)

# freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
sgd = SGD(learning_rate=0.03, momentum=0.9, name = "sgd")
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

# print the summary of the model
model.summary()
#--------------------------------------------------#



# model callbacks
early = EarlyStopping(monitor="val_accuracy", min_delta = 0.0001, patience=20, mode="auto")
checkpoint = ModelCheckpoint("tmp/checkpoint", monitor="val_accuracy", save_best_only=True, save_weights_only=False, mode="auto")
rlrop = ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, min_delta=0.0001, patience=15, mode="auto")

# train the model
model.fit(X_train, Y_train, batch_size = 128, epochs = 200, validation_data=(X_test, Y_test), callbacks=[early, checkpoint, rlrop])

predsTrain = model.evaluate(X_train, Y_train)
predsTest = model.evaluate(X_test, Y_test)
print("Training Accuracy: ", predsTrain[1])
print("Testing Accuracy: ", predsTest[1])



