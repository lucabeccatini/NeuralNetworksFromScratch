import numpy as np
import matplotlib.pyplot as plt

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

    # ! ! ! use softmax activation for the output layer
    # feed-forward 
    # the results are arrays: L, which contains all the layers values (needed for prop), and y, which are the outputs
    def forward(self, x):
        L = np.copy(x)
        for l in range(self.l_n):
            if l == 0:
                H = self.activation(np.dot(L, self.w[l]) + self.b[l])            # H contains the values of the layers l+1
                L = np.vstack([L, H])                                            # add H to L as a new row
            else:
                if l == (self.l_n - 1):
                    y = self.activation(np.dot(L[l], self.w[l]) + self.b[l])
                else:
                    H = self.activation(np.dot(L[l], self.w[l]) + self.b[l])
                    L = np.vstack([L, H])
        return L, y

    # ! ! ! use loss = cross entropy
    # back-propagation
    # use the chain rule to find the derivative of the loss function respect the weights and the bias
    # the loss function is the mean squared error
    def propagation(self, L, y, y_train):                                        # L, y, y_train are arrays
        for l in reversed(range(self.l_n)): 
            if l == (self.l_n - 1):
                b_err = (y_train - y) * y * (np.ones(self.l_s) - y)              # biases errors 
                w_err = np.outer(L[l], b_err)                                    # weights errors
                # outer(a, b): a defines the row, b the column. example: w_iJ = L[l]_i * b_j
            else: 
                b_err = np.dot(b_err, w_err) * L[l] * (np.ones(self.l_s) - L[l]) 
                w_err = np.outer(L[l], b_err)
            self.b[l] += b_err * self.lr                                         
            self.w[l] += w_err * self.lr 

    
    # training
    def training(self, X_train, Y_train):
        n_train = len(X_train)
        n_epochs = self.epochs 
        #print("\n\nThe initial weights and bias are: \nw = {},\n b = {}".format(self.w.T, self.b))
        for i in range(n_epochs):
            for j in range(n_train):
                x = np.copy(X_train[j])
                y_train = np.copy(Y_train[j])
                L, y = self.forward(x)
                self.propagation(L, y, y_train)
                #print("\nThe weights and bias after {}/{} training samples are: \nw = {}, b = {}".format(j, n_train, self.w.T, self.b))
            #if i % (n_epochs/1000) == 0:
                #print("\n\nThe weights and bias after {}/{} epochs are: \nw = {},\n b = {}".format(i+1, self.epochs, self.w.T, self.b))
        print("\n\nThe weights and bias after {} epochs are: \nw = {},\n b = {}".format(self.epochs, self.w.T, self.b))
    
    
    # predictng function
    def predict(self, x):
        L, y = self.forward(x)
        return y


    # function to plot the NN decision regions
    def plot_decision_regions(self, X_train, y_data, res=0.1):                 # only for 2d input
        x_min = X_train[:, 0].min() - 0.5                                        # x and y limit for the grid
        x_max = X_train[:, 0].max() + 0.5 
        y_min = X_train[:, 1].min() - 0.5
        y_max = X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))   # create a grid that covers the samples
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        Y_pred = np.empty([len(X_grid), self.l_s]) 
        Z_pred = np.zeros(len(X_grid)) 
        for m in range(len(X_grid)):  
            Y_pred[m] = self.predict(X_grid[m])                                 # predicted outputs over the grid
            #print("Y[{}] = {}".format(m, Y_pred[m]))
            for n in range(self.l_s): 
                Z_pred[m] += (n+1) * (Y_pred[m][n] > 0.5).astype(int)          # needed a single value per point for contourf
            #print(Z_pred[m]) 
        Z_pred = np.reshape(Z_pred, xx.shape)
        lev = [0, 1, 2, 3]
        plt.contourf(xx, yy, Z_pred, levels = lev, alpha = 0.5)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.scatter(X_train[:, 0], X_train[:, 1], c = y_data,  alpha = 0.2)     # .. c=Y_true.reshape(-1),  alpha=0.2)
        plt.title("Decision region plot of NN with {} layers and {} layers size".format(self.l_n, self.l_s)) 
        
        