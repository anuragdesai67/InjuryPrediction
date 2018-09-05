# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:00:40 2018

@author: anurag
"""

# Importing libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten, Input, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import losses
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def encoding_train_test(y_train,y_test):
    label_encoder = LabelEncoder()
    integer_encoded_train = label_encoder.fit_transform(y_train)
    integer_encoded_test = label_encoder.fit_transform(y_test)
# =============================================================================
#     le = preprocessing.LabelEncoder()
#     le.fit(y_train)
#     print(list(le.classes_))
#     train_y = le.transform(train_y)
# =============================================================================
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_train = integer_encoded_train.reshape(len(integer_encoded_train), 1)
    integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
    onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded_train)
    onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded_test)
    
    return onehot_encoded_train, onehot_encoded_test, integer_encoded_train, integer_encoded_test
   

def create_model(embedding_dim,act_fun):
    model = Sequential()
    model.add(Dense(9, input_dim=embedding_dim, activation=act_fun))
    model.add(Dense(10, activation=act_fun))
    model.add(Dense(8, input_dim=embedding_dim, activation=act_fun))
    model.add(Dense(5, activation='softmax'))
    return model

# Compile model
def compile_model(model, error_function):
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=error_function, optimizer=adam, metrics=['accuracy'])
    return model

# Training the model
def train_model(model,num_epochs,b_size,X_train,y_train_encoded):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    import time
    start = time.clock()
    model.fit(X_train, y_train_encoded, validation_split = 0.2,epochs=num_epochs, batch_size=b_size, callbacks=[early_stopping])
    print (time.clock() - start)
    
# Evaluate the model
def evaluate_model(model,X_test,y_test_encoded):
    scores = model.evaluate(X_test, y_test_encoded)
    return scores


def import_train_data(filename):
    train_file = pd.read_csv(filename, header=1)
    train_x = train_file.iloc[:, 0:9].values
    train_y = train_file.iloc[:, 9].values
    return train_x,train_y

def get_normed_mean_cov(X):
    X_std = StandardScaler().fit_transform(X)
    X_mean = np.mean(X_std, axis=0)
    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0]-1)
    return X_std, X_mean, X_cov

X_train,y_train = import_train_data("new_york_data_balanced.csv")
#Stratified sampling for test and train data
# =============================================================================
# Meta = pd.read_csv("data.csv",skip_blank_lines=True);
# y = Meta.pop("Original Disability Type")
# X = Meta
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
X_train, _, _ = get_normed_mean_cov(X_train)
X_test, _, _ = get_normed_mean_cov(X_test)
# =============================================================================
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.fit_transform(X_test)
# =============================================================================
#one hot encoding
y_train_encoded, y_test_encoded, y_integer_train, y_integer_test = encoding_train_test(y_train, y_test)

#neural network
# Training parameters
embedding_dim = 9
batch_size = 128
num_epochs = 30     
model = create_model(embedding_dim,K.relu)

print("------------------Sum of Squares error---------------------------------------------------------------------------")
# Sum of Squares Error
model = compile_model(model,losses.categorical_crossentropy)
train_model(model,num_epochs,batch_size,X_train,y_train_encoded)
scores = evaluate_model(model,X_test,y_test_encoded)
print("\nOverall ClassificationAccuracy: %.2f%%" % (scores[1]*100))
print("\n")
print("-----------------TRAINING DATA---------------------------------------------------------------------------")
y_pred = model.predict_classes(X_train)
print(classification_report(y_integer_train, y_pred))
print(confusion_matrix(y_integer_train, y_pred))
print(accuracy_score(y_integer_train, y_pred))
print("-----------------TESTING DATA---------------------------------------------------------------------------")
y_pred = model.predict_classes(X_test)
print(classification_report(y_integer_test, y_pred))
print(confusion_matrix(y_integer_test, y_pred))
print(accuracy_score(y_integer_test, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_integer_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_integer_test)))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_integer_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_integer_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_integer_test, y_pred))
print('True positive = ', confusion_matrix[0][0])
print('False positive = ', confusion_matrix[0][1])
print('False negative = ', confusion_matrix[1][0])
print('True negative = ', confusion_matrix[1][1])

# =============================================================================
# from sklearn.neural_network import MLPClassifier
# 
# classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
# classifier.fit(X_train,y_train_encoded)
# 
# y_pred_1 = classifier.predict(X_test)
# print (precision_score(y_test_encoded, y_pred_1))
# print(confusion_matrix(y_test_encoded, y_pred_1))
# print(accuracy_score(y_test_encoded, y_pred_1, normalize=False))
# print(y_pred_1)
# print(y_test_encoded)
# =============================================================================

