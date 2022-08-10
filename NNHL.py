import numpy as np

class NeuralNetwork():
    
    # instantiate class parameters
    # consider only NN with equal layers size
    def __init__(self, layers_number, layers_size, epochs = 10000, lr = 0.1, seed = 7):
        np.random.seed(seed)                                                     # seed
        self.l_n = layers_number                                                 # number of layers 
        self.l_s = layers_size                                                   # size of the layers 
        self.b = np.random.rand(layers_number, layers_size)                      # bias matrix 
        self.w = np.random.rand(layers_number, layers_size, layers_size)         # weights array of matrix 
        self.lr = lr                                                             # learning rate 
        self.epochs = epochs                                                     # number of epochs 


    # activation function
    # consider only Sigmoid activation function
    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    
    # feed-forward 
    # the results are: L, which contains all the layers values (needed for prop), and Y, which are the outputs
    def forward(self, x):
        L = np.copy(x)
        for l in range(self.l_n):
            if l == 0:
                H = self.activation(np.dot(L, self.w[l]) + self.b[l])            # H contains the values of the layers l+1
                L = np.vstack([L, H])                                            # add H to L as a new row
            if l == (self.l_n - 1):
                y = self.activation(np.dot(L[l], self.w[l]) + self.b[l])
            else:
                H = self.activation(np.dot(L[l], self.w[l]) + self.b[l])
                L = np.vstack([L, H])
        return L, y

    
    # back-propagation
    # use the chain rule to find the derivative of the loss function respect the weights and the bias
    # the loss function is the mean squared error
    def propagation(self, L, y, y_true):
        for l in reversed(range(self.l_n)):
            if l == (self.l_n - 1):
                b_err = (y_true - y) * y * (np.ones(self.l_s) - y)               # biases errors 
                w_err = np.outer(L[l], b_err)                                    # weights errors
                # outer(a, b): a defines the row, b the column. example: w_iJ = L[l]_i * b_j
            else: 
                b_err = np.dot(b_err, w_err) * L[l] * (np.ones(self.l_s) - L[l]) 
                w_err = np.outer(L[l], b_err)
            self.b[l] += b_err * self.lr                                         
            self.w[l] += w_err * self.lr 

    
    # training
    def training(self, X_train, Y_true):
        n_train = len(X_train)
        n_epochs = self.epochs
        for i in range(n_epochs):
            #print("\n\nThe weights and bias after {}/{} epochs are: \nw = {}, b = {}".format(i, self.epochs, self.w.T, self.b))
            for j in range(n_train):
                x = np.copy(X_train[j])
                y_true = np.copy(Y_true[j])
                L, y = self.forward(x)
                self.propagation(L, y, y_true)
                #print("\nThe weights and bias after {}/{} training samples are: \nw = {}, b = {}".format(j, n_train, self.w.T, self.b))
   
    
    # predictng function
    def predict(self, x):
        L, y = self.forward(x)
        return y
