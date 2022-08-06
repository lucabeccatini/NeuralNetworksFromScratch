import numpy as np



class Perceptron():
    
    # instantiate input, weights and bias
    def __init__(self, size):
        np.random.seed(42)
        self.size = size                    # number of inputs
        self.w = np.random.rand(size, 1)    # weights
        self.b = np.random.rand(1)          # bias
        
    """
    Do i want to save training X,Y
    # load training dataset
    def get_train_dataset(self, X_train, Y_train):
        self.X_train = X_train              # training input
        self.Y_train = Y_train              # training output
        # !!! chech the shape
        pass
    """
    
    # activation function
    # consider only Sigmoid activation function
    def activation(z):
        return 1 / (1 + np.exp(-z))
    
    # feed-forward
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return self.activation(z)
    
    # back-propagation
    def propagation(self, x, y_true):
        w_err = x * (y_true - self.forward(x)) * self.activation(x) * (1 - self.activation(x))  
        self.w += w_err
    