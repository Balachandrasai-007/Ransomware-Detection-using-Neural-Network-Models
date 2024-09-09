# Importing Required Libraries for the below code 
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. 
pd.read_csv) 
import importlib 
import os 
import tensorflow as tf 
from keras.models import Model 
from keras.layers import Input, Conv2D,MaxPooling2D, 
Dropout,Dense,Flatten, Concatenate, BatchNormalization 
from keras.activations import relu 
from keras.optimizers import Adam 
from keras.metrics import CategoricalAccuracy, 
Recall,Precision,F1Score 
from keras.losses import CategoricalCrossentropy 
from keras.applications import DenseNet121 
from keras.applications.inception_v3 import InceptionV3 
from keras.preprocessing import image_dataset_from_directory 
import warnings 
from keras import utils 
import keras.utils as plot_model 
import matplotlib.pyplot as plt 
from keras.callbacks import CSVLogger 
importlib.reload(tf) 
warnings.filterwarnings('ignore') 
#train_datset_dir = "D:/malimg-20240330T120722Z-001/malimg" 
train_datset_dir="C:/Users/surya/Downloads/archive/malimg_dataset"
input_shape = (75,75,3) 
num_classes = len(os.listdir(train_datset_dir)) 
batch_size = 32 
epochs = 10 
def get_datagen(): 
 
train_datagen=tf.keras.preprocessing.image.ImageDataGenerator( 
 featurewise_center=False, 
 samplewise_center=False, 
 featurewise_std_normalization=False, 
 samplewise_std_normalization=False, 
 zca_whitening=False, 
 zca_epsilon=1e-06, 
 rotation_range=0, 
 width_shift_range=0.0, 
 height_shift_range=0.0, 
 brightness_range=None, 
 shear_range=0.0, 
 zoom_range=0.0, 
 channel_shift_range=0.0, 
 fill_mode 
='nearest', 
 cval=0.0, 
 horizontal_flip=False, 
 vertical_flip=False, 
 rescale=1/255, 
 preprocessing_function=None, 
 data_format=None, 
 validation_split=0.2, 
interpolation_order=1, 
 dtype=None 
) 
 #train_datagen = 
tf.keras.preprocessing.image.ImageDataGenerator( 
 # rotation_range=20, 
 # horizontal_flip=True, 
 # vertical_flip=False, 
 # rescale=1 / 255, 
 # validation_split=0.2 
 #) 
 train_generator = 
train_datagen.flow_from_directory(train_datset_dir, 
 target_size= 
(input_shape[0],input_shape[1]), 
 batch_size=32, 
 subset='training') 
 validation_generator = 
train_datagen.flow_from_directory(train_datset_dir, 
 target_size= 
(input_shape[0],input_shape[1]), 
 batch_size=32, 
 subset='validation') 
 return train_generator, validation_generator 
train_generator, validation_generator = get_datagen()
# inputs = Input(input_shape) 
inception = InceptionV3(include_top=False,weights='imagenet',input_shape = input_shape) 
for layers in inception.layers: 
 layers.trainable = False 
dense = Flatten()(inception.output) 
dense = Dense(256)(dense) 
dense = Dense(64)(dense) 
dense = Dense(3)(dense) 
model = Model(inputs = inception.inputs,outputs = [dense]) 
model.summary() 
model.compile(optimizer=Adam(), 
 
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
 metrics=[CategoricalAccuracy(), Recall()]) 
num_classes = len(os.listdir(train_datset_dir)) 
print("Number of subdirectories in train dataset: ", num_classes) 
# Fit the model with concise arguments 
model.fit(train_generator, 
 validation_data=validation_generator) 
model.evaluate(validation_generator) 
#DenseNet121 
dn = DenseNet121(weights='imagenet',include_top=False,input_shape = input_shape) 
for layers in dn.layers: 
 layers.trainable = False 
dense = Flatten()(dn.output) 
dense = Dense(256)(dense) 
dense = Dense(64)(dense) 
dense = Dense(3)(dense) 
model = Model(inputs = dn.inputs,outputs = [dense]) 
model.summary() 
model.compile(optimizer = Adam(), 
 
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
 metrics=[CategoricalAccuracy(), Recall(),Precision()]) 
model.fit(train_generator,validation_data = validation_generator,epochs = epochs) 
model.evaluate(validation_generator) 
#Ensemble 
dn = DenseNet121(weights='imagenet', include_top=False,input_shape = input_shape) 
inception = InceptionV3(include_top=False,weights='imagenet',input_shape = input_shape) 
inputs = Input(shape = input_shape) 
incept_out = inception(inputs) 
dn_out = dn(inputs)
incept_out = Flatten()(incept_out) 
dn_out = Flatten()(dn_out) 
conv = Concatenate(axis = 1)([incept_out,dn_out]) 
conv = Dense(512)(conv) 
conv = Dropout(0.3)(conv) 
conv = Dense(256)(conv) 
conv = BatchNormalization()(conv) 
conv = Dense(128)(conv) 
conv = Dense(64)(conv) 
conv = Dense(3)(conv) 
model = Model(inputs = [inputs],outputs=[conv]) 
model.summary() 
model.compile(optimizer = 
Adam(),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=
True),metrics=[CategoricalAccuracy(), Recall(),Precision()]) 
csv_logger = CSVLogger('training_metrics.log', append=True, 
separator=';') 
# Train the model with the CSVLogger callback 
model.fit(train_generator, validation_data=validation_generator, 
callbacks=[csv_logger]) 
# Fit the model with callbacks 
#model.fit(train_generator,validation_data=validation_generator,e
pochs=epochs,callbacks=[plotter]) 
df = pd.read_csv("D:/VS Code Programs/training_metrics.log", sep=';')
# Plot the data 
plt.figure(figsize=(8, 10)) 
# Plot each metric except 'epoch' 
for metric in df.columns[1:]: 
 plt.plot(df['epoch'], df[metric], label=metric) 
# Add labels and title 
plt.xlabel('Epoch') 
plt.ylabel('Value') 
plt.title('Training Metrics Over Epochs') 
plt.ylim(0.6000,1.0000) 
# Add a legend 
plt.legend() 
# Show the plot 
plt.show()