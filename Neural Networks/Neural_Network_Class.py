import numpy as np

class Neural_Network(object):
    """docstring for Neural_Network"""
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize  = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, input):
        # Propagate inputs through network
        self.z2          = np.dot(input, self.W1)
        self.activation2 = self.sigmoid(self.z2)
        self.z3          = np.dot(self.activation2, self.W2)
        output           = self.sigmoid(self.z3)
        return output

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))


NN     = Neural_Network()
input  = [[0, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1]]
output = np.array([[0, 1, 1, 0]]).T
print NN.forward(input)
