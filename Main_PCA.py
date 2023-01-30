from scipy.io import loadmat
import pandas as pd
import numpy as np
import keras
import scipy
import scipy.io
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.models import Sequential
import tensorflow as tf
import sklearn.metrics
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Dropout

#--------------------PCA with Percent------------------------------------
def trasform(per, img):
    per = per / 100
    pca_ = PCA(n_components=per).fit(img)
    transformed = pca_.transform(img)
    proj = pca_.inverse_transform(transformed)
    return proj

#---------------------------------------------------------
def plot(people, img):
    img = img / 255
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(people, cmap='gray')
    ax[1].imshow(img , cmap='gray')
    plt.show()

#---------------------------------------------------------
def Load_Data():

    mat_01 = scipy.io.loadmat('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Pattern_Classification_pro/P_02/female.mat')
    mat_01 = {k: v for k, v in mat_01.items() if k[0] != '_'}

    array_01 = []
    people_female = []
    for i in range(0, 200):
        s = mat_01['female'][:, i]
        s1 = np.array(s)
        s1 = s1.reshape(48, 60)
        array_01.append(s1)

        img = array_01[i]
        img = img / 255
        people_female.append(trasform(30, img))

        plot(people_female[i], img) #Plot male image and male image PCA

    #--------------------------------
    mat_02 = scipy.io.loadmat('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Pattern_Classification_pro/P_02/male.mat')
    mat_02 = {k:v for k, v in mat_02.items() if k[0] != '_'}

    array_02 = []
    people_male = []
    for i in range(0, 200):
        s = mat_02['male'][:, i]
        s2 = np.array(s)
        s2 = s2.reshape(48, 60)
        array_02.append(s2)

        img = array_02[i]
        img = img / 255
        people_male.append(trasform(30, img))

        plot(people_male[i], img) #Plot male image and male image PCA


    female_male = np.concatenate((people_female, people_male))
    print(female_male.shape)

    x_female_male = np.zeros((len(female_male), 48, 60))
    y_female_male = np.zeros(len(female_male))

    y_female = np.zeros(len(array_01))
    y_male = np.zeros(len(array_02))


    c1=0
    for i in range(0, len(array_01)):#Female
        y_female[c1] = 0
        c1=c1+1

    c2 = 0
    for i in range(0, len(array_02)):#male
        y_male[c2] = 1
        c2=c2+1

    y_female_male = np.concatenate((y_female, y_male)) #Concat 2Array
    c = 0
    for i in range(0, len(female_male)):
        x_female_male[c,:,:] = female_male[i]
        c=c+1


    return x_female_male,y_female_male

#======================================================================
z = Load_Data()
x = z[0]
y = z[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


num_classes = 2
y_test = to_categorical(y_test, num_classes)
y_train = to_categorical(y_train, num_classes)

# ===========continue===================
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

#X_train /= 255
#X_test /= 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=3, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1

acc_per_fold = []
loss_per_fold = []

for train, test in kfold.split(inputs, targets):
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(48, 60, 1)))  
    model.add(Conv2D(64, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 1))  # default stride is 2
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 1))  # default stride is 2
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # ===================================
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    print(f'Training for fold {fold_no} ...')

    model.fit(inputs[train], targets[train], batch_size=64, epochs=5)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
