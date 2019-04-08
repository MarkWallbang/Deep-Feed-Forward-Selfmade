# Dataset source: http://onlinestatbook.com/2/case_studies/sat.html
# Source of DFF-Programming Information: https://enlight.nyc/projects/neural-network/
# Book: https://www.deeplearningbook.org/

import numpy as np
import pandas as pd

# Klasse des Neural Networks definieren
class  Neural_Network(object):
    def __init__(self):
        self.inputCount = 2
        self.outputCount = 1
        self.hiddenCount = 3

        # initiate random weights
        self.weights1 = np.random.randn(self.inputCount, self.hiddenCount)
        self.weights2 = np.random.randn(self.hiddenCount, self.outputCount)

    def forward(self, X):
        self.z = np.dot(X, self.weights1)  # dot product of X (input) and first set of 2x3 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.weights2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o

    # activation function
    def sigmoid (self,s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propagate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(
            self.weights2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error

        self.weights1 += X.T.dot(self.z2_delta)  # adjusting first set (input --> hidden) weights
        self.weights2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


# Setup data from csv (Independent Variables : Highschool Grade & verbal Grade in HS
# Dependant Variable = University Grade)
data = pd.read_csv("sat.csv", delimiter=";")
X = data[["high_GPA","verb_SAT"]]
Y = data[["univ_GPA"]]


# Maximalwert von "high_GPA", um später denormalisieren zu können
max1 = np.amax(data.iloc[:,0],axis=0)

# Maximalwert von "verb_SAT", um später denormalisieren zu können
max2 = np.amax(data.iloc[:,2],axis=0)


# scale units to (0:1) range
X = X/np.amax(X,axis=0)
Y = Y/np.amax(Y,axis=0)

# Erstellen eines Neuronalen Netzes
NN = Neural_Network()

# Trainieren des NN durch 300 Iterationen
for i in range(300):
  print("Input: \n" + str(X))
  print("Actual Output: \n" + str(Y))
  print("Predicted Output: \n" + str(NN.forward(X)))
  print("Loss: \n" + str(np.mean(np.square(Y - NN.forward(X)))))
  print("\n")
  NN.train(X, Y)

# Bestimmen eines neuen Wertes (newdata)
newdata = np.array([2.1/max1,594/max2])
print("Universitätsnote: " + str(NN.forward(newdata)*max1))

