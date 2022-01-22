#%% Week 2 activity - Naive Bayes Classifier

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import sklearn.datasets as skl_datasets
import sklearn.naive_bayes as skl_nb
import time
import pandas as pd

n_classes = 2
n_samples = 100

def split_partition(X, y, train_split):
    # merge y (classifications) and X (co-ordinates)
    y = np.reshape(y, (len(y), 1))
    Xy = np.concatenate((X, y), axis=1)
    
    # split dataset into training and test partitions
    train_split = 0.7
    sample_size = Xy.shape[0]
    indices = np.random.permutation(sample_size)
    i = int(sample_size * train_split) 
    train_indices, test_indices = indices[:i], indices[i:]
    Xy_train, Xy_test = Xy[train_indices, :], Xy[test_indices, :]

    return Xy_train, Xy_test

def split_colour(Xy, n_classes):
    # split co-ordinates and classification arrays
    X, y = Xy[:, :n_classes], Xy[:, n_classes]

    # prefill empty arrays for co-ordinates
    X_green, X_blue, y_green, y_blue = [], [], [], []

    # define arrays of green and blue co-ordinates
    for idx, classification in enumerate(y):
        if y[idx] == 0:
            X_green.append(X[idx, :])
            y_green.append(y[idx])
        elif y[idx] == 1:
            X_blue.append(X[idx, :])
            y_blue.append(y[idx])

    return np.array(X_green), np.array(X_blue)

class Partition:
    def __init__(self, Xy, n_classes):
        print('Creating partition class')
        X_green, X_blue = split_colour(Xy, n_classes)
        self.Green_X = X_green
        self.Blue_X = X_blue

def compareClassifications(y_both) :
    comparison = []
    for i in np.arange(y_both.shape[0]):
        if y_both[i, 0] == 0:
            if y_both[i, 1] == 0:
                comparison.append('TN')
            if y_both[i, 1] == 1:
                comparison.append('FP')
        if y_both[i, 0] == 1:
            if y_both[i, 1] == 0:
                comparison.append('FN')
            if y_both[i, 1] == 1:
                comparison.append('TP')
    TP_total, FP_total = comparison.count('TP'), comparison.count('FP')
    TN_total, FN_total = comparison.count('TN'), comparison.count('FN')
    T_e = abs(TP_total+TN_total+FP_total+FN_total)
    print(TP_total, FP_total, TN_total, FN_total)
    return comparison, TP_total, FP_total, TN_total, FN_total, T_e

def safeDiv(num, den):
    try:
        return num/den
    except ZeroDivisionError:
        return 0

# def get_results(y, y_pred):


# generate two-class dataset (green and blue)
X, y = skl_datasets.make_blobs(n_samples = n_samples, centers = 2,
 n_features=n_classes, random_state=0)
# ... where X is the co-ordinates 
#     and Y gives their classification (green=0 & blue=1)

# print('Before all values made positive, X = \n', X, '\n')

# remove -ve values from x and y components of X 
# ... by adding smallest component to all
X[:] = X[:] + abs(np.min(X[:]))
# print('After all values made negative, X = \n', X, '\n')

# split dataset into training and testing partitions
train_split = 0.7
train_all, test_all = split_partition(X, y, train_split)

# formally classify colours (not using Naive-Bayes)
Train = Partition(train_all, n_classes)
Test = Partition(test_all, n_classes)

# create Naive Bayes classifier model
train_all_X = train_all[:, :n_classes]
train_all_y = train_all[:, n_classes]
time_start = time.time()
nb_model = skl_nb.CategoricalNB().fit(train_all_X, train_all_y)
time_total = time.time() - time_start

# make predictions
test_all_X = test_all[:, :n_classes]
test_all_y = test_all[:, n_classes]
y_pred = nb_model.predict(test_all_X)

#%% merge NB-predicted y (classifications) and X (co-ordinates)
y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
Xy_pred = np.concatenate((test_all_X, y_pred), axis=1)    

# create separate arrays for NB-classified co-ordinates
X_green_pred, X_blue_pred = split_colour(Xy_pred, n_classes)

# Performance Metrics:
print('PERFORMANCE METRICS')

# Compare predicted and actual classifications
y_both = np.concatenate((test_all_y.reshape(len(test_all_y), 1), y_pred), axis=1)

comparison, TP_total, FP_total, TN_total, FN_total, T_e = compareClassifications(y_both)

y_both_pd = pd.DataFrame(y_both, columns = ['Actual Classification', 'Naive-Bayes Predicted'])
y_both_pd['Classification Vs Prediction Comparison'] = comparison

print(y_both_pd)


#%% time taken to train model
print('\nTime taken to train model = ', time_total, 'seconds\n')

# classifier accuracy = (TP+TN)/|Te|
accuracy = safeDiv(TP_total+TN_total, T_e)
print('Classification Accuracy = ', accuracy)

# precision = TP / (TP + FP)
precision = safeDiv(TP_total, TP_total + FP_total)
print('Classification Precision = ', precision)

# recall = TP / (TP + FN)
recall = safeDiv(TP_total, TP_total + FN_total)
print('Classification Recall = ', recall)

# # receiver operating characteristics (ROC) curve


# plot original generated training data
fig1 = plt.figure
plt.title('Train Dataset - Actual Classification')
plt.scatter(Train.Green_X[:,0], Train.Green_X[:,1], c='green', alpha = 0.6, s=4)
plt.scatter(Train.Blue_X[:,0],  Train.Blue_X[:,1],  c='blue',  alpha = 0.6, s=4)
plt.legend(['green (train)', 'blue (train)'], loc='lower left' )

# plot original generated testing data
fig2, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('Test Dataset - Actual Classification')
ax1.scatter(Test.Green_X[:,0],  Test.Green_X[:,1],  c='green', alpha = 0.6, s=4)
ax1.scatter(Test.Blue_X[:,0],   Test.Blue_X[:,1],   c='blue',  alpha = 0.6, s=4)
ax1.legend(['green (test)', 'blue (test)'], loc='lower left' )
ax1.axis([0, np.amax(X[:,0]), 0, np.amax(X[:,1])])

# plot Naive-Bayes predicted test results
ax2.set_title('Test Dataset - Naive-Bayes Predicted Classification')
ax2.scatter(X_green_pred[:,0],  X_green_pred[:,1],  c='green', alpha = 0.6, s=4)
ax2.scatter(X_blue_pred[:,0],   X_blue_pred[:,1],   c='blue',  alpha = 0.6, s=4)
ax2.axis([0, np.amax(X[:,0]), 0, np.amax(X[:, 1])])
ax2.legend(['green (test)', 'blue (test)'], loc='lower left' )
plt.show()
#%%