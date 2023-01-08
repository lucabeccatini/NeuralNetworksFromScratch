import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    # instantiate class parameters
    # consider only NN with equal layers size
    def __init__(self, layers, seed=7):
        np.random.seed(seed)                                                     # seed
        self.lr = 0.01                                                           # learning rate 
        self.epochs = 100                                                          # number of epochs

        self.layers = layers                                                     # layers strucure
        n_bias = np.sum([layers[i] for i in range(1, len(layers))])          # number of total bias of nn
        n_weights = np.sum([layers[i]*layers[i+1] for i in range(len(layers)-1)])    # number of total weights of nn

        self.b = np.zeros(n_bias)                                          # bias array
        self.w = np.zeros(n_weights)
        self.n = np.zeros(layers[0]+n_bias)                                         # nodes values of last prediction, used for the back-propagation
        self.b_error = np.zeros(n_bias)                                          # corrections for biases of last prediction, used for the back-propagation
        self.w_error = np.zeros(n_weights)                                           # corrections for weights of last prediction, used for the back-propagation
        # weights initialization
        for i in range(1, len(self.layers)):
            # heuristic 
            self.w[self.ind_w([i, 0, 0]):self.ind_w([i+1, 0, 0])] = np.random.normal(0, np.sqrt(2/self.layers[i-1]), self.ind_w([i+1, 0, 0])-self.ind_w([i, 0, 0])) 
            # Xavier 
            #self.w[self.ind_w([i, 0, 0]):self.ind_w([i+1, 0, 0])] = np.random.normal(0, 1/self.layers[i-1], self.ind_w([i+1, 0, 0])-self.ind_w([i, 0, 0]))
        self.activation = ['relu', 'sigmoid']
        self.loss_train, self.loss_val = np.zeros(0), np.zeros(0)        


    # index of bias
    def ind_b(self, ind):
        # ind[0]: layer index; ind[1]: node index
        if (len(ind)==2):
            if (ind[0]==1):
                ind_l = 0
            elif (ind[0]>1):
                ind_l = np.sum([self.layers[i] for i in range(1, ind[0])])               # sum of nodes in the previous layers
            ind_b = ind_l + ind[1]
            return ind_b
        ###### add break error here


    # index of weights
    def ind_w(self, ind):
        # ind[0]: layer index of the final node; ind[1]: initial node index; ind[2]: final node index
        if (len(ind)==3):
            if (ind[0]==1):
                ind_w = ind[1]*self.layers[ind[0]] + ind[2]
            elif (ind[0]==len(self.layers)):     # to avoid error self.layer[len(self.layers)]
                ind_w = np.sum([self.layers[i]*self.layers[i+1] for i in range(len(self.layers)-1)]) 
            else:
                ind_l = np.sum([self.layers[i]*self.layers[i+1] for i in range(ind[0]-1)])           # sum of nodes in the previous layers
                ind_w = ind_l + ind[1]*self.layers[ind[0]] + ind[2]
            return ind_w


    # index of node
    def ind_n(self, ind):
        # ind[0]: layer index; ind[1]: node index
        if (len(ind)==2):
            if (ind[0]==0):
                ind_l = 0
            elif (ind[0]==-1):
                ind_l = np.sum([self.layers[i] for i in range(len(self.layers)-1)])               # sum of nodes in the previous layers
            elif (ind[0]>0): 
                ind_l = np.sum([self.layers[i] for i in range(ind[0])])               # sum of nodes in the previous layers
            ind_n = ind_l + ind[1]
            return ind_n


    def bias(self, ind): 
        # ind[0]: layer index; ind[1]: node index
        if (len(ind)==1):
            if (ind[0]==1):
                ind_l = 0
            elif (ind[0]>1):
                ind_l = np.sum([self.layers[i] for i in range(1, ind[0])])               # sum of nodes in the previous layers
            ind_l2 = ind_l + self.layers[ind[0]]
            b = self.b[ind_l:ind_l2]
            return b 
        else:
            if (ind[0]==1):
                ind_l = 0
            elif (ind[0]>1):
                ind_l = np.sum([self.layers[i] for i in range(1, ind[0])])               # sum of nodes in the previous layers
            ind_b = ind_l + ind[1]
            b = self.b[ind_b]
            return b


    def weight(self, ind):
        # ind[0]: layer index; ind[1]: initial node index; ind[2]: final node index
        if (len(ind)==1):
            if (ind[0]==1):
                ind_l = 0
            elif (ind[0]>1):
                ind_l = np.sum([self.layers[i]*self.layers[i+1] for i in range(ind[0]-1)])           # sum of nodes in the previous layers
            ind_l2 = ind_l + self.layers[ind[0]-1]*self.layers[ind[0]]
            w = self.w[ind_l:ind_l2]
            w = np.reshape(w, (self.layers[ind[0]],self.layers[ind[0]-1]))
            return w
        if (len(ind)==2):
            if (ind[0]==1):
                ind_l = 0
            elif (ind[0]>1):
                ind_l = np.sum([self.layers[i]*self.layers[i+1] for i in range(ind[0]-1)])           # sum of nodes in the previous layers
            ind_w1 = ind_l + ind[1]*self.layers[ind[0]-1]
            ind_w2 = ind_w1 + self.layers[ind[0]-1]
            w = self.w[ind_w1:ind_w2]
            return w
        if (len(ind)==3):
            if (ind[0]==1):
                ind_l = 0
            elif (ind[0]>1):
                ind_l = np.sum([self.layers[i]*self.layers[i+1] for i in range(ind[0]-1)])           # sum of nodes in the previous layers
            ind_w = ind_l + ind[1]*self.layers[ind[0]-1] + ind[2]
            w = self.w[ind_w]
            return w


    def node(self, ind):
        # ind[0]: layer index; ind[1]: node index
        # node[0]=inputs; node[i]=hidden layers nodes; node[-1]=outputs
        ind_l = 0
        if (len(ind)==1):
            if (ind[0]==-1):
                ind_l = np.sum([self.layers[i] for i in range(len(self.layers)-1)])               # sum of nodes in the previous layers
            if (ind[0]>0):
                ind_l = np.sum([self.layers[i] for i in range(ind[0])])               # sum of nodes in the previous layers
            ind_l2 = ind_l + self.layers[ind[0]]
            n = self.n[ind_l:ind_l2]
        elif (len(ind)==2):
            if (ind[0]==-1):
                ind_l = np.sum([self.layers[i] for i in range(len(self.layers)-1)])               # sum of nodes in the previous layers
            if (ind[0]>0):
                ind_l = np.sum([self.layers[i] for i in range(ind[0])])               # sum of nodes in the previous layers
            ind_n = ind_l + ind[1]
            n = self.n[ind_n]
        return n        


    def bias_error(self, ind): 
        # ind[0]: layer index; ind[1]: node index
        if (len(ind)==1):
            if (ind[0]==1):
                ind_l = 0
            else:
                ind_l = np.sum([self.layers[i] for i in range(1, ind[0])])               # sum of nodes in the previous layers
            ind_l2 = ind_l + self.layers[ind[0]]
            b_err = self.b_error[ind_l:ind_l2]
            return b_err 
        else:
            if (ind[0]==1):
                ind_l = 0
            else:
                ind_l = np.sum([self.layers[i] for i in range(1, ind[0])])               # sum of nodes in the previous layers
            ind_b = ind_l + ind[1]
            b_err = self.b_error[ind_b]
            return b_err


    def weight_error(self, ind):
        # ind[0]: layer index; ind[1]: initial node index; ind[2]: final node index
        if (len(ind)==1):
            if (ind[0]==1):
                ind_l = 0
            else:
                ind_l = np.sum([self.layers[i]*self.layers[i+1] for i in range(ind[0]-1)])           # sum of nodes in the previous layers
            ind_l2 = ind_l + self.layers[ind[0]-1]*self.layers[ind[0]]
            w_err = self.w_error[ind_l:ind_l2]
            w_err = np.reshape(w_err, (self.layers[ind[0]],self.layers[ind[0]-1]))
            return w_err
        if (len(ind)==2):
            if (ind[0]==1):
                ind_l = 0
            else:
                ind_l = np.sum([self.layers[i]*self.layers[i+1] for i in range(ind[0]-1)])           # sum of nodes in the previous layers
            ind_w1 = ind_l + ind[1]*self.layers[ind[0]-1]
            ind_w2 = ind_w1 + self.layers[ind[0]-1]
            w_err = self.w_error[ind_w1:ind_w2]
            return w_err
        if (len(ind)==3):
            if (ind[0]==1):
                ind_l = 0
            else:
                ind_l = np.sum([self.layers[i]*self.layers[i+1] for i in range(ind[0]-1)])           # sum of nodes in the previous layers
            ind_w = ind_l + ind[1]*self.layers[ind[0]-1] + ind[2]
            w_err = self.w_error[ind_w]
            return w_err



    # activation function
    # available: sigmoid, relu, linear
    def activation_func(self, z, i):
        if (i==len(self.layers)-1):
            if (self.activation[1]=='sigmoid'):
                res = 1 / (1 + np.exp(-z))
            elif (self.activation[1]=='relu'):
                res = np.maximum(np.zeros_like(z), z)
            elif (self.activation[1]=='linear'):
                res = z
        else:
            if (self.activation[0]=='sigmoid'):
                res = 1 / (1 + np.exp(-z))
            elif (self.activation[0]=='relu'):
                res = np.maximum(np.zeros_like(z), z)
            elif (self.activation[0]=='linear'):
                res = z
        return res


    # partial derivative of the activation function respect the pre-activation node
    # available: sigmoid, relu, linear
    def der_activation(self, n, i):
        # n: value of the node after the activation function; i: layer index
        if (i==len(self.layers)-1):
            if (self.activation[1]=='sigmoid'):
                res = n * (np.ones_like(n) - n)
            elif (self.activation[1]=='relu'):
                res = np.heaviside(n, 0)
            elif (self.activation[1]=='linear'):
                res = np.ones_like(n)
        else:
            if (self.activation[0]=='sigmoid'):
                res = n * (np.ones_like(n) - n)
            elif (self.activation[0]=='relu'):
                res = np.heaviside(n, 0)
            elif (self.activation[0]=='linear'):
                res = np.ones_like(n)
        return res


    # loss function
    # consider only mse loss function
    def loss_func(self, y_train):
        if (self.layers[-1]==1):
            mse = (y_train - self.node([-1, 0]))**2
        else:
            mse = 0
            for i in range(self.layers[-1]):
                mse += (y_train[i] - self.node([-1, i]))**2
            mse = mse/len(self.layers)
        return mse


    # partial derivative of the loss function respect the output layer
    # consider only mse loss function
    def der_loss(self, y_train):
        #der = 2 * (y_train - self.node([-1]))
        der = 2 * (self.node([-1]) - y_train)
        return der


    # predictng function
    def predict(self, x_data):
        self.forward(x_data)
        return self.node([-1])


    # feed-forward 
    # the results are arrays: L, which contains all the hiddane layers nodes values (needed for propagation), and y, which are the outputs
    def forward(self, X):
        self.n[0:self.ind_n([1, 0])] = X 
        for i in range(1, len(self.layers)):
            self.n[self.ind_n([i, 0]):self.ind_n([i+1, 0])] = self.activation_func( np.sum(self.node([i-1]) * self.weight([i]), axis=1) + self.bias([i]), i ) 
        return


    # back-propagation
    # use the chain rule to find the corrections to weights and bias
    # the loss function is the mean squared error
    def propagation(self, y_train):                                        # L, y, y_train are arrays
        for i in reversed(range(1, len(self.layers))): 
            if (i==(len(self.layers)-1)):
                self.b_error[self.ind_b([i, 0]):self.ind_b([i+1, 0])] = self.der_loss(y_train) * self.der_activation(self.node([i]), i)     # biases errors 
                w_err = np.outer(self.bias_error([i]), self.node([i-1]))       # weights errors
                self.w_error[self.ind_w([i, 0, 0]):self.ind_w([i+1, 0, 0])] = np.reshape(w_err, (self.ind_w([i+1, 0, 0])-self.ind_w([i, 0, 0])))
                # outer(a, b): a defines the row, b the column. example: w_iJ = L[l]_i * b_j. Outer result is a matrix, [:, 0] to have an array
            else: 
                self.b_error[self.ind_b([i, 0]):self.ind_b([i+1, 0])] = np.dot(self.bias_error([i+1]), self.weight([i+1])) * self.der_activation(self.node([i]), i)      # biases errors 
                w_err = np.outer(self.bias_error([i]), self.node([i-1]))                        # weights errors
                self.w_error[self.ind_w([i, 0, 0]):self.ind_w([i+1, 0, 0])] = np.reshape(w_err, (self.ind_w([i+1, 0, 0])-self.ind_w([i, 0, 0])))
        return 
 

    # training and evaluation
    def fit(self, X_train, Y_train, X_val, Y_val, epochs, batch_size=100, learn_rate=0.01, patience=10, min_improvement=0, shuffle=True):
        self.loss_train, self.loss_val = np.zeros(epochs), np.zeros(epochs)        
        n_train = len(X_train)
        n_val = len(X_val)
        self.epochs = epochs
        self.lr = learn_rate
        n_batch = n_train // batch_size
        not_improved = 0                                             # counter to interrupt training
        i_best = 0                                                   # to save the best model
        corr_w = np.zeros(len(self.w))                               # to add the corrections to weights at the end of the batch
        corr_b = np.zeros(len(self.b))                               # to add the corrections to biases at the end of the batch

        if (shuffle==True):
            Y_train = np.reshape(Y_train, (len(Y_train), 1))
            XY_train = np.hstack((X_train, Y_train))

        for i_epoch in range(epochs):                                # loop for each epoch
            if (shuffle==True):
                np.random.shuffle(XY_train)
                X_train = XY_train[:, :-1]
                Y_train = XY_train[:, -1]

            if (i_epoch>=52):
                a = 0

            for i_n_batch in range(n_batch+1):                        # loop for each batch
                if (i_n_batch < n_batch):
                    for i_batch in range(batch_size):                # loop for each event inside batch
                        x_train = np.copy(X_train[i_n_batch*batch_size+i_batch])
                        y_train = np.copy(Y_train[i_n_batch*batch_size+i_batch])
                        self.forward(x_train) 
                        self.propagation(y_train) 
                        corr_w += self.w_error 
                        corr_b += self.b_error 
                        self.loss_train[i_epoch] += self.loss_func(y_train) 
                elif (i_n_batch == n_batch):
                    for i_batch in range(n_train % batch_size):
                        x_train = np.copy(X_train[i_n_batch*batch_size+i_batch])
                        y_train = np.copy(Y_train[i_n_batch*batch_size+i_batch])
                        self.forward(x_train)
                        self.propagation(y_train)
                        corr_w += self.w_error
                        corr_b += self.b_error
                        self.loss_train[i_epoch] += self.loss_func(y_train) 
                ##### corr_w and b need to be renormalized???
                self.w -= corr_w * self.lr
                self.b -= corr_b * self.lr
                corr_w, corr_b = 0, 0
            for i_val in range(n_val):
                self.predict(X_val[i_val])
                y_val =  np.copy(Y_val[i_val])
                self.loss_val[i_epoch] += self.loss_func(y_val)
            self.loss_train[i_epoch] = self.loss_train[i_epoch] / n_train
            self.loss_val[i_epoch] = self.loss_val[i_epoch] / n_val
            print("Epoch: {}/{}  ,  training loss: {}  ,  validation loss: {}".format(i_epoch+1, epochs, self.loss_train[i_epoch], self.loss_val[i_epoch]))
            if ((self.loss_val[i_epoch-1]-self.loss_val[i_epoch]) > min_improvement):
                #not_improved = 0
                if(self.loss_val[i_epoch] < self.loss_val[i_best]):
                    wgt_best = self.w                                # to save the best weights
                    i_best = i_epoch
            else:
                not_improved += 1
                if (not_improved >= patience):
                    self.w = wgt_best
                    print("\nTraining early stopped")
                    break 
        #plt.plot(self.loss_train, color='blue', label="train loss")
        #plt.plot(self.loss_val, color='red', label="validation loss")
        #plt.title('model loss')
        #plt.ylabel('loss')
        #plt.xlabel('epoch')
        #plt.legend(['Train', 'Validation'], loc='best')
        #plt.savefig("/home/lb_linux/NeuralNetworksFromScratch/train_NNFS.pdf")




