import numpy as np



class Perceptron():
    
    # instantiate class parameters
    def __init__(self, size):
        np.random.seed(42)
        self.size = size                    # number of inputs
        self.w = np.random.rand(size, 1)    # weights
        self.b = np.random.rand(1)          # bias
        self.lr = 0.05                      # learning rate
        self.epochs = 10                    # number of epochs


    """    
    # load training dataset
    def get_train_dataset(self, X_train, Y_true):
        self.X_train = X_train              # training input
        self.Y_true = Y_true              # true output of the training samples
        # ! ! ! chech the shape
        print("The training inputs are {} \n The true value of the training outputs are {}".format(X_train, Y_true))
        # ! ! ! do i want to save and then empty X_train, Y_train ? ? ?
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
            print("\n\nThe weights and bias after {}/{} epochs are: w = {}, b = {}".format(i, self.epochs, self.w, self.b))
            for j in n_train:
                x = np.copy(X_train[j, :])
                y_true = np.copy(Y_true[j])
                y = self.forward(x)
                self.propagation(x, y, y_true)
                print("\nThe weights and bias after {}/{} training samples are: w = {}, b = {}".format(j, n_train, self.w, self.b))
        

    # predicting function
    def predict(self, x):
        y = self.forward(x)
        print("\nThe prediction for {} is {}".format(x, y))
        return y    