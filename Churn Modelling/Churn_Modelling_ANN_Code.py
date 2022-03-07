
####################################   CHURN MODELLING   #########################################################  


#Data Preprocessing

#Importing the libraries
import pandas as pd
import numpy as np

#Reading files
dataset=pd.read_csv('D:\Interview Preparation/11.Self Projects/Deep Learning Projects/Churn Modelling/Churn_Modelling.txt')

#Dependent and Independent variables
len(dataset.columns)

X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]

#Create dummy variables
geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)
X=pd.concat([X,geography,gender],axis=1)

#Drop unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

#Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Creating ANN Model 
from keras.models import Sequential

classifier=Sequential() #Empty neural network

#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))
#Units- Hidden neurons
#For 'relu' activation function the weight initialization we uses are called 'he_unifrom'
#input_dim-input baggage

#Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the training set
model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=100)

# list all data in history

print(model_history.history.keys())

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Predicting and Evaluating the model
#Predicting the test set result
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)
print('Accuracy : ',accuracy)
