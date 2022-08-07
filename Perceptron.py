import numpy as np



class Perceptron():
    
    # instantiate class parameters
    def __init__(self, size):
        np.random.seed(42)
        self.size = size                    # number of inputs
        self.w = np.random.rand(size, 1)    # weights
        self.b = np.random.rand(1)          # bias
        self.lr = 0.05                      # learning rate
        self.epochs = 100                   # number of epochs
        
    """
    Do i want to save training X,Y ???
    # load training dataset
    def get_train_dataset(self, X_train, Y_true):
        self.X_train = X_train              # training input
        self.Y_true = Y_true              # true output of the training samples
        # ! ! ! chech the shape
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
    def propagation(self, x, y, y_true):
        w_err = x * (y_true - y) * y * (1 - y)  
        self.w += w_err * self.lr
    
    # training
    def training(self, X_train, Y_true):
        n_train = len(X_train)
        for i in self.epochs:
            for j in n_train:
                x = np.copy(X_train[j, :])
                y_true = np.copy(Y_true[j])
                y = self.forward(x)
                self.propagation(x, y, y_true)                

    # predicting function
    def predict(self, x):
        y = self.forward(x)
        return y    