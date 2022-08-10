import numpy as np

class NeuralNetwork():
    
    # instantiate class parameters
    # consider only NN with equal layers size
    def __init__(self, layers_number, layers_size, epochs = 10000, lr = 0.1, seed = 7):
        np.random.seed(seed)                                                     # seed
        self.l_n = layers_number                                                 # number of layers
        self.l_s = layers_size                                                   # size of the layers
        self.b = np.random.rand(layers_number, layers_size)                    # bias
        self.w = np.random.rand([layers_number][layers_size, layers_size])       # weights
        self.lr = lr                                                             # learning rate
        self.epochs = epochs                                                     # number of epochs


    # activation function
    # consider only Sigmoid activation function
    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    
    # feed-forward 
    # the results are: L, which contains all the layers values (needed for prop), and Y, which are the outputs
    def forward(self, X):
        L = np.copy(X)
        for l in range(self.l_n):
            if l == 0:
                H = self.activation(np.dot(L, self.w[l]) + self.b[l])
                L = np.vstack([L, H])
            if l == self.l_n:
                Y = self.activation(np.dot(L[l], self.w[l]) + self.b[l])
            else:
                H = self.activation(np.dot(L[l], self.w[l]) + self.b[l])
                L = np.vstack([L, H])
        return L, Y
