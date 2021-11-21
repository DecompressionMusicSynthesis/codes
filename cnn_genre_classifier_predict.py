import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os,shutil

#文件路径
DATA_PATH = "D:\python and c++\srtp\data_2.json"
MULTI_PATH="D:\data\softmusic\multi"
SINGLE_PATH="D:\data\softmusic\single"
ORIGIN_PATH="D:\python and c++\srtp\softmusic"
PREDICT_PATH="D:\python and c++\srtp\data_ini.json"

#加载MFCC数据
def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

#加载需要预测的特征数据
def load_toBePred(data_path):
    """Loads predict dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X0 (ndarray): data needs to be predicted
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    files=np.array(data["filename"])
    # y = np.array(data["labels"])
    return X,files

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["acc"], label="train accuracy")
    axs[0].plot(history.history["val_acc"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
                              #第一个维度是样本的个数
    X_validation = X_validation[..., np.newaxis]   
    X_test = X_test[..., np.newaxis]    #在外面再加一层[]，相当于是增加了维度

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates CNN model

    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    #create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))   #Dense是全连接层
    model.add(keras.layers.Dropout(0.3))

    # output layer
                                #10 for 10 classes
    model.add(keras.layers.Dense(2, activation='softmax'))

    return model


def predict(model, X, filename):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    
    #根据数据的预测标签将数据放入对应的文件夹中
    if predicted_index==0:
        shutil.copyfile(ORIGIN_PATH+"\\"+filename,MULTI_PATH+"\\"+filename)
    else:
        shutil.copyfile(ORIGIN_PATH+"\\"+filename,SINGLE_PATH+"\\"+filename)
    return predicted_index

if __name__ == "__main__":

    # get train, validation, test splits
                                                                                #test/validation
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model                                                                       #将样本以32位单位切分
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy on test set is:{}".format(test_acc))
    print('\nTest accuracy:', test_acc)

    # pick a sample to predict from the test set
    # X_to_predict = X_test[100]
    # y_to_predict = y_test[100]

    # predict sample
    # predict(model, X_to_predict, y_to_predict)

    #save model
    model.save('path_to_saved_model')
    print("saved total model")

    #load model
    # new_model = keras.models.load_model('path_to_saved_model')
    X_pred,files=load_toBePred(PREDICT_PATH)
    X_pred = X_pred[..., np.newaxis]
    print(X_pred[0].shape)   #130*13
    resRe=[]
    for i in range(X_pred.shape[0]):
        res=predict(model,X_pred[i],files[i*10])
        resRe.append([i+1,res])
