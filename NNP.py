import numpy as np



class Perceptron():
    
    # instantiate class parameters
    def __init__(self, size, epochs = 10000, lr = 0.1, seed = 7):
        np.random.seed(seed)                # seed
        self.size = size                    # number of inputs
        self.w = np.random.rand(size)       # weights
        self.b = np.random.rand(1)          # bias
        self.lr = lr                        # learning rate
        self.epochs = epochs                # number of epochs


    """    
    # load training dataset
    # ! ! ! do i want to save and then empty X_train, Y_train ? ? ?
    def get_train_dataset(self, X_train, Y_true):
        self.X_train = X_train              # training input
        self.Y_true = Y_true              # true output of the training samples
        # ! ! ! chech the shape
        print("The training inputs are {} \n The true value of the training outputs are {}".format(X_train, Y_true))
    """
    
    # activation function
    # consider only Sigmoid activation function
    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    
    # feed-forward
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return self.activation(z)

    
    # back-propagation
    # use the chain rule to find the derivative of the loss function respect the weights and the bias
    # the loss function is the mean squared error
    def propagation(self, x, y, y_true):
        b_err = (y_true - y) * y * (1 - y)   # bias correction
        w_err = x * b_err                    # weights corrections
        self.w += w_err * self.lr
        self.b += b_err * self.lr

    
    # training
    def training(self, X_train, Y_true):
        n_train = len(X_train)
        n_epochs = self.epochs
        for i in range(n_epochs):
            #print("\n\nThe weights and bias after {}/{} epochs are: \nw = {}, b = {}".format(i, self.epochs, self.w.T, self.b))
            for j in range(n_train):
                x = np.copy(X_train[j])
                y_true = np.copy(Y_true[j])
                y = self.forward(x)
                self.propagation(x, y, y_true)
                #print("\nThe weights and bias after {}/{} training samples are: \nw = {}, b = {}".format(j, n_train, self.w.T, self.b))
        

    # predicting function
    def predict(self, x):
        y = self.forward(x)
        print("\nThe prediction for {} is {}".format(x, y))
        return y    