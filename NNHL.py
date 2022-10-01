import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    # instantiate class parameters
    # consider only NN with equal layers size
    def __init__(self, layers_number, layers_size, lr=0.01, seed=7):
        np.random.seed(seed)                                                     # seed
        self.l_n = layers_number                                                 # number of layers 
        self.l_s = layers_size                                                   # size of the layers 
        self.b = np.random.rand(layers_number, layers_size)                      # bias matrix 
        self.w = np.random.rand(layers_number, layers_size, layers_size)         # weights array of matrix 
        self.lr = lr                                                             # learning rate 
        self.epochs = 0                                                          # number of epochs 
        self.w_error = np.zeros(shape=self.w.shape)                              # used for the back-propagation
        self.b_error = np.zeros(shape=self.b.shape)                              # used for the back-propagation

    # activation function
    # consider only Sigmoid activation function
    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    # loss function
    # consider only mse loss function
    def loss(self, y_true, y_pred):
        mse = np.zeros(1)
        for i in range(self.l_s):
            mse += (y_true[i] - y_pred[i])**2
        mse = mse/self.l_s
        return mse

    # derivative of the loss function
    def der_loss(self, a, y_pred):
        der = a * y_pred * (np.ones(self.l_s) - y_pred)
        return der

    # predictng function
    def predict(self, x_data):
        L, y = self.forward(x_data)
        return y

    # feed-forward 
    # the results are arrays: L, which contains all the layers values (needed for propagation), and y, which are the outputs
    def forward(self, x_data):
        L= np.zeros((self.l_n, len(x_data)))
        L[0] = x_data 
        for l in range(self.l_n):
                if (l < (self.l_n - 1)):
                    L[l+1] = self.activation(np.dot(L[l], self.w[l]) + self.b[l])
                else:
                    y_pred = self.activation(np.dot(L[l], self.w[l]) + self.b[l])
        return L, y_pred
 
    # back-propagation
    # use the chain rule to find the corrections to weights and bias
    # the loss function is the mean squared error
    def propagation(self, L, y_pred, y_train):                                        # L, y, y_train are arrays
        for l in reversed(range(self.l_n)): 
            if l == (self.l_n - 1):
                b_err = self.der_loss(y_train - y_pred, y_pred)      # biases errors 
                w_err = np.outer(L[l], b_err)                        # weights errors
                # outer(a, b): a defines the row, b the column. example: w_iJ = L[l]_i * b_j
            else: 
                b_err = self.der_loss(np.dot(b_err, w_err), L[l]) 
                w_err = np.outer(L[l], b_err)
            self.b_error[l] = b_err                                          
            self.w_error[l] = w_err 
        return  


    # training and evaluation
    def learn(self, X_train, Y_train, X_val, Y_val, epochs, batch_size=1, patience=3, min_improvement=0):
        n_train = len(X_train)
        n_val = len(X_val)
        self.epochs = epochs
        n_batch = n_train // batch_size
        not_improved = 0                                             # counter to interrupt training
        loss_train, loss_val = np.zeros(epochs), np.zeros(epochs)        
        i_best = 0
        corr_w = np.zeros(shape=self.w.shape)
        corr_b = np.zeros(shape=self.b.shape)
        for i_epoch in range(epochs):                                # loop for each epoch
            for i_n_batch in range(n_batch):                         # loop for each batch
                if (i_n_batch < (n_batch-1)):
                    for i_batch in range(batch_size):                # loop for each event inside batch
                        x_train = np.copy(X_train[i_n_batch*batch_size+i_batch])
                        y_train = np.copy(Y_train[i_n_batch*batch_size+i_batch])
                        L, y_pred = self.forward(x_train)
                        self.propagation(L, y_pred, y_train)    
                        corr_w += self.w_error
                        corr_b += self.b_error
                        loss_train[i_epoch] += self.loss(y_train, y_pred) 
                if (i_n_batch == (n_batch-1)):
                    for i_batch in range(n_train % batch_size):
                        x_train = np.copy(X_train[i_n_batch*batch_size+i_batch])
                        y_train = np.copy(Y_train[i_n_batch*batch_size+i_batch])
                        L, y_pred = self.forward(x_train)
                        self.propagation(L, y_pred, y_train)
                        corr_w += self.w_error
                        corr_b += self.b_error
                        loss_train[i_epoch] += self.loss(y_train, y_pred)                     
                self.w += corr_w * self.lr
                self.b += corr_b * self.lr
            for i_val in range(n_val):
                y_pred_val = self.predict(X_val[i_val])
                y_val =  np.copy(Y_val[i_val])
                loss_val[i_epoch] += self.loss(y_val, y_pred_val)
            loss_train[i_epoch] = loss_train[i_epoch] / n_train
            loss_val[i_epoch] = loss_val[i_epoch] / n_val
            print("Epoch: {}/{}  ,  training loss: {}  ,  validation loss: {}".format(i_epoch+1, epochs, loss_train[i_epoch], loss_val[i_epoch]))
            if ((loss_val[i_epoch-1]-loss_val[i_epoch]) > min_improvement):
                not_improved = 0
                if(loss_val[i_epoch] < loss_val[i_best]):
                    wgt_best = self.w                                # to save the best weights
                    i_best = i_epoch
            else:
                not_improved += 1
                if (not_improved >= patience):
                    self.w = wgt_best
                    print("\nTraining early stopped")
                    break 
        plt.plot(loss_train, color='blue', label="train loss")
        plt.plot(loss_val, color='red', label="validation loss")
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.show()


    # plot the NN decision regions
    def plot_decision_regions(self, X_data, y_labels, res=0.1):                 # only for 2d input
        x_min = X_data[:, 0].min() - 0.5                                        # x and y limit for the grid
        x_max = X_data[:, 0].max() + 0.5 
        y_min = X_data[:, 1].min() - 0.5
        y_max = X_data[:, 1].max() + 0.5
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
        plt.scatter(X_data[:, 0], X_data[:, 1], c = y_labels,  alpha = 0.2)     # .. c=Y_true.reshape(-1),  alpha=0.2)
        plt.title("Decision region plot of NN with {} layers and {} layers size".format(self.l_n, self.l_s)) 
        
