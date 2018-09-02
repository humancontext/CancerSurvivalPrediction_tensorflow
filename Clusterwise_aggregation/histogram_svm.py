import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('D://data//1-10//data.csv')
feature_dict = {}
type_dict = {}

for i in range(len(df)):
    feature_dict[df.loc[i, 'ImageIndex']] = df.loc[i, '5yrs']
    type_dict[df.loc[i, 'ImageIndex']] = df.loc[i, 'data_type']


prediction_dir = 'D:\\data\\1-10\\predictions\\logs_nobn_PhC5_2\\'

list_image = os.listdir(prediction_dir)

x_train = []
y_train = []
voting_train = []

x_val = []
y_val = []

# 1: +1, 0: -1
voting_val = []

x_test = []
y_test = []
voting_test = []


for image in list_image:
    image_index = int(image[:4])
    path_prediction = os.path.join(prediction_dir, image)
    predictions = np.load(path_prediction)

    histogram, bin_edges = np.histogram(predictions,
                                        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        density=True)
    #print(histogram)
    label = feature_dict[image_index]
    if type_dict[image_index] == 'train':

        vote = 0
        for prediction in predictions:
            if prediction > 0.5:
                vote += 1
            else:
                vote -= 1
        if vote >= 0:
            voting_train.append(1)
        else:
            voting_train.append(0)
        x_train.append(histogram)
        y_train.append(label)
    elif type_dict[image_index] == 'val':

        vote = 0
        for prediction in predictions:
            if prediction > 0.5:
                vote += 1
            else:
                vote -= 1
        if vote >= 0:
            voting_val.append(1)
        else:
            voting_val.append(0)
        x_val.append(histogram)
        y_val.append(label)
    else:
        ax = sns.distplot(predictions, bins=20)
        ax.set_title('Image(%d) Label = %d' % (image_index, label))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10)
        plt.show()
        vote = 0
        for prediction in predictions:
            if prediction > 0.5:
                vote += 1
            else:
                vote -= 1
        if vote >= 0:
            voting_test.append(1)
        else:
            voting_test.append(0)
        x_test.append(histogram)
        y_test.append(label)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_val = np.asarray(x_val)
y_val = np.asarray(y_val)

clf = SVC()
clf.fit(X=x_train, y=y_train)
y_pre = clf.predict(x_test)

print(classification_report(y_true=y_test, y_pred=y_pre))
print(confusion_matrix(y_true=y_test, y_pred=y_pre))