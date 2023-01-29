from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.io import loadmat
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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



    female_male = np.concatenate((array_01, array_02))

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


x_train = np.array(x_train)
n_samples = len(x_train)
x_train = x_train.reshape((n_samples, -1))

x_test = np.array(x_test)
n_samples = len(x_test)
x_test = x_test.reshape((n_samples, -1))

#----------------LDA classification---------------------------------
lda = LDA(n_components=1)
x_train = lda.fit_transform(x_train, y_train)

y_pred = lda.predict(x_test)

MA = metrics.accuracy_score(y_test, y_pred)
print("MA is :")
print(MA)
print("#================")

CM = confusion_matrix(y_test, y_pred)
DetaFrame_cm = pd.DataFrame(CM, range(2), range(2))
sns.heatmap(DetaFrame_cm, annot=True)
plt.show()

print("classification_report is :")
print(classification_report(y_test, y_pred))
