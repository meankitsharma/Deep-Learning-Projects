#Keras Tuner- Decide Number of Hidden Layers And Neuron In Neural Network

############################################# Air Quality Index #####################################################

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

df=pd.read_csv('D:/Interview Preparation/11.Self Projects/Deep Learning Projects/Keras_Tuner- Air Quality Index Problem/Air_Quality_Index_Data.txt')

X=df.iloc[:,:-1] #Independent features
y=df.iloc[:,-1] #Dependent feature

#Hyperparameters
#1.How many number of hidden layers we should have?
#2.How many number of neurons we should have in hidden layers?
#3.Learning Rate

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):                    #Neural network will have layers 2 to 20
        model.add(layers.Dense(units=hp.Int('units_' + str(i),     #Inside each layer we can use 32 to 512 nuerons,Dense function is used to create neurons
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  #Adding one output layer,which will output 1 values as it is a regresion problem
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='project',
    project_name='Air Quality Index')

tuner.search_space_summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tuner.search(X_train, y_train,epochs=5,validation_data=(X_test, y_test))

tuner.results_summary()

