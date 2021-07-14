#!/usr/bin/env python
# coding: utf-8
import os
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend

#Set seed, so that every time you run, you will get the same (reproducible) results.
SEED_VALUE = 0
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.compat.v1.set_random_seed(SEED_VALUE)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Dataset is taken from https://www.kaggle.com/c/lish-moa/data
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

train_features.drop(['sig_id'], axis='columns', inplace=True)

train_features.loc[:, 'cp_type'] = train_features.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
train_features.loc[:, 'cp_dose'] = train_features.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

test_features.loc[:, 'cp_type'] = test_features.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
test_features.loc[:, 'cp_dose'] = test_features.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

train_targets_scored.drop(['sig_id'], axis='columns', inplace=True)

train_features_values = train_features.values
train_targets_values = train_targets_scored.values

input_dimension = train_features_values.shape[1]
output_dimension = train_targets_values.shape[1]

input_units = input_dimension
UNITS_DENSE_LAYER_1 = 128
UNITS_DENSE_LAYER_2 = 64
output_units = output_dimension

inputs = keras.Input(shape=input_units)
dense_layer1 = layers.Dense(UNITS_DENSE_LAYER_1, activation="relu", name = "dense_layer1")(inputs)
dropout_layer1 = layers.Dropout(0.5, name = "dropout_layer1")(dense_layer1)
batch_normalization_layer1 = layers.BatchNormalization(name = "batch_normalization_layer1")(dropout_layer1)

dense_layer2 = layers.Dense(UNITS_DENSE_LAYER_2, activation="relu", name = "dense_layer2")(batch_normalization_layer1)
dropout_layer2 = layers.Dropout(0.5, name = "dropout_layer2")(dense_layer2)
batch_normalization_layer2 = layers.BatchNormalization(name = "batch_normalization_layer2")(dropout_layer2)

outputs = layers.Dense(output_units, activation="sigmoid", name = "Output_Layer")(batch_normalization_layer2)

model = keras.Model(inputs=inputs, outputs=outputs, name="moa_model")

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=["accuracy"],
)

model.fit(train_features, train_targets_values, batch_size=100, epochs=75)

model.summary()
keras.utils.plot_model(model, show_shapes = True)

# Save the model (Serialization)
model.save('neural_network_model.h5', save_format='h5')

column_names=sample_submission.columns

test_id=x_test=test_features['sig_id'].values
x_test=test_features.drop(['sig_id'], axis='columns', inplace=False)

df = pd.DataFrame(x_test)

y_pred=model.predict(x_test)

id_list= list(test_id)
y_pred_list= list(y_pred)

dictionary={}
for column in column_names:
    dictionary[column]=[]

column_names = list(column_names)
column_names.remove('sig_id')

dictionary['sig_id'] = test_id
for i in range(len(y_pred)):
    for j in range(len(column_names)):
        dictionary[column_names[j]].append(y_pred[i][j])

df=pd.DataFrame(dictionary)

df.to_csv('./submission.csv',index=False)
