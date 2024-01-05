# importing the required libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# check version on sklearn
print('Version of sklearn:', sklearn.__version__)

# loading the pre-processed dataset
data = pd.read_csv('/Users/paramanandbhat/Downloads/Solving loan prediction challenge using neural network in keras/loan_prediction_data.csv')

# looking at the first five rows of the dataset
data.head()

print(data.head())


# checking missing values
print(data.isnull().sum())

# checking the data type
print(data.dtypes)

# removing the loan_ID since these are just the unique values
data = data.drop('Loan_ID', axis=1)

# looking at the shape of the data
print(data.shape)


# separating the independent and dependent variables

# storing all the independent variables as X
X = data.drop('Loan_Status', axis=1)

# storing the dependent variable as y
y = data['Loan_Status']

# shape of independent and dependent variables
X.shape, y.shape

print(X.shape, y.shape)

## 2. Creating training and validation set
# Creating training and validation set

# stratify will make sure that the distribution of classes in train and validation set it similar
# random state to regenerate the same train and validation set
# test size 0.2 will keep 20% data in validation and remaining 80% in train set

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=data['Loan_Status'],random_state=10,test_size=0.2)

# shape of training and validation set
(X_train.shape, y_train.shape), (X_test.shape, y_test.shape)

print((X_train.shape, y_train.shape), (X_test.shape, y_test.shape))

## 3. Defining the architecture of the model
# checking the version of keras
import keras
print(keras.__version__)

# checking the version of tensorflow
import tensorflow as tf
print(tf.__version__)

### a. Create a model

# importing the sequential model
from keras.models import Sequential

### b. Defining different layers
# importing different layers from keras
from keras.layers import InputLayer, Dense 

# number of input neurons
X_train.shape

print(X_train.shape)

# number of features in the data
X_train.shape[1]

print(X_train.shape[1])

# defining input neurons
input_neurons = X_train.shape[1]

# number of output neurons
# since loan prediction is a binary classification problem, we will have single neuron in the output layer 

# define number of output neurons
output_neurons = 1

# number of hidden layers and hidden neurons

# It is a hyperparameter and we can pick the hidden layers and hidden neurons on our own

# define hidden layers and neuron in each layer
number_of_hidden_layers = 2
neuron_hidden_layer_1 = 10
neuron_hidden_layer_2 = 5

# activation function of different layers

# for now I have picked relu as an activation function for hidden layers, you can change it as well
# since it is a binary classification problem, I have used sigmoid activation function in the final layer

# defining the architecture of the model
model = Sequential()
model.add(InputLayer(input_shape=(input_neurons,)))
model.add(Dense(units=neuron_hidden_layer_1, activation='relu'))
model.add(Dense(units=neuron_hidden_layer_2, activation='relu'))
model.add(Dense(units=output_neurons, activation='sigmoid'))

# summary of the model
model.summary()
print(model.summary())

# number of parameters between input and first hidden layer

print(input_neurons*neuron_hidden_layer_1)

# number of parameters between input and first hidden layer

# adding the bias for each neuron of first hidden layer

print(input_neurons*neuron_hidden_layer_1 + 10)

# number of parameters between first and second hidden layer

print(neuron_hidden_layer_1*neuron_hidden_layer_2 + 5)

# number of parameters between second hidden and output layer

print(neuron_hidden_layer_2*output_neurons + 1)

## 4. Compiling the model (defining loss function, optimizer)

# compiling the model

# loss as binary_crossentropy, since we have binary classification problem
# defining the optimizer as adam
# Evaluation metric as accuracy

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

## 5. Training the model
# training the model

# passing the independent and dependent features for training set for training the model

# validation data will be evaluated at the end of each epoch

# setting the epochs as 50

# storing the trained model in model_history variable which will be used to visualize the training process

model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

## 6. Evaluating model performance on validation set
# getting predictions for the validation set
# Get predictions for the test set
raw_predictions = model.predict(X_test)

# Convert predictions to class labels
# Assuming it's a classification problem with categorical outputs
prediction = np.argmax(raw_predictions, axis=1)


print(prediction)

# calculating the accuracy on validation set
accuracy_score(y_test, prediction)
print(accuracy_score(y_test, prediction))

### Visualizing the model performance
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# summarize history for accuracy
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
