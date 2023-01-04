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

        #self.b = np.random.rand(n_bias)                                          # bias array
        self.b = np.zeros(n_bias)                                          # bias array
        self.w = np.random.randn(n_weights)                                           # weights array 
        self.n = np.zeros(layers[0]+n_bias)                                         # nodes values of last prediction, used for the back-propagation
        self.b_error = np.zeros(n_bias)                                          # corrections for biases of last prediction, used for the back-propagation
        self.w_error = np.zeros(n_weights)                                           # corrections for weights of last prediction, used for the back-propagation


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
    # consider only Sigmoid activation function
    def activation(self, z):
        res = 1 / (1 + np.exp(-z))
        return res

    # loss function
    # consider only mse loss function
    def loss(self, y_true):
        if (self.layers[-1]==1):
            mse = (y_true - self.node([-1, 0]))**2
        else:
            mse = 0
            for i in range(self.layers[-1]):
                mse += (y_true[i] - self.node([-1, i]))**2
            mse = mse/len(self.layers)
        return mse

    # derivative of the loss function
    def der_loss(self, diff, nodes):
        der = diff * nodes * (np.ones(self.layers[-1]) - nodes)
        return der

    # predictng function
    def predict(self, x_data):
        self.forward(x_data)
        return self.node([-1])


    # feed-forward 
    # the results are arrays: L, which contains all the hiddane layers nodes values (needed for propagation), and y, which are the outputs
    def forward(self, X):
        self.n[0:self.ind_n([1, 0])] = X   ###### modify not only the first of the first layer
        for i in range(1, len(self.layers)):
            self.n[self.ind_n([i, 0]):self.ind_n([i+1, 0])] = self.activation( np.sum(self.node([i-1]) * self.weight([i]), axis=1) + self.bias([i]) ) 
        return

        """
            for j in range(self.layers[i]):
                if (i==1):
                    self.node[1, j] = self.activation( np.sum(X*self.weight[0, j]) + self.bias[1, j] )
                elif (i==(len(self.layers)-1)):
                    self.y[j] = self.activation( np.sum(self.node[i-1] * self.weight[i-1, j]) + self.bias[i, j] )
                else:
                    self.node[i, j] = self.activation( np.sum(self.node[i-1] * self.weight[i-1, j]) + self.bias[i, j] )
        """


    # back-propagation
    # use the chain rule to find the corrections to weights and bias
    # the loss function is the mean squared error
    def propagation(self, y_train):                                        # L, y, y_train are arrays
        for i in reversed(range(1, len(self.layers))): 
            if (i==(len(self.layers)-1)):
                self.b_error[self.ind_b([i, 0]):self.ind_b([i+1, 0])] = 2 * self.der_loss(y_train - self.node([i]), self.node([i]))      # biases errors 
                w_err = np.outer(self.bias_error([i]), self.node([i-1]))       # weights errors
                self.w_error[self.ind_w([i, 0, 0]):self.ind_w([i+1, 0, 0])] = np.reshape(w_err, (self.ind_w([i+1, 0, 0])-self.ind_w([i, 0, 0])))
                # outer(a, b): a defines the row, b the column. example: w_iJ = L[l]_i * b_j. Outer result is a matrix, [:, 0] to have an array
            else: 
                self.b_error[self.ind_b([i, 0]):self.ind_b([i+1, 0])] = self.der_loss(np.dot(self.bias_error([i+1]), self.weight_error([i+1])), self.node([i]))      # biases errors 
                w_err = np.outer(self.bias_error([i]), self.node([i-1]))                        # weights errors
                self.w_error[self.ind_w([i, 0, 0]):self.ind_w([i+1, 0, 0])] = np.reshape(w_err, (self.ind_w([i+1, 0, 0])-self.ind_w([i, 0, 0])))
        return 
 

    # training and evaluation
    def fit(self, X_train, Y_train, X_val, Y_val, epochs, batch_size=100, learn_rate=0.01, patience=10, min_improvement=0):
        n_train = len(X_train)
        n_val = len(X_val)
        self.epochs = epochs
        self.lr = learn_rate
        n_batch = n_train // batch_size
        not_improved = 0                                             # counter to interrupt training
        loss_train, loss_val = np.zeros(epochs), np.zeros(epochs)        
        i_best = 0                                                   # to save the best model
        corr_w = np.zeros(len(self.w))                               # to add the corrections to weights at the end of the batch
        corr_b = np.zeros(len(self.b))                               # to add the corrections to biases at the end of the batch

        for i_epoch in range(epochs):                                # loop for each epoch
            for i_n_batch in range(n_batch+1):                        # loop for each batch
                if (i_n_batch < n_batch):
                    for i_batch in range(batch_size):                # loop for each event inside batch
                        x_train = np.copy(X_train[i_n_batch*batch_size+i_batch])
                        y_train = np.copy(Y_train[i_n_batch*batch_size+i_batch])
                        self.forward(x_train) 
                        self.propagation(y_train) 
                        corr_w += self.w_error 
                        corr_b += self.b_error 
                        loss_train[i_epoch] += self.loss(y_train) 
                elif (i_n_batch == n_batch):
                    for i_batch in range(n_train % batch_size):
                        x_train = np.copy(X_train[i_n_batch*batch_size+i_batch])
                        y_train = np.copy(Y_train[i_n_batch*batch_size+i_batch])
                        self.forward(x_train)
                        self.propagation(y_train)
                        corr_w += self.w_error
                        corr_b += self.b_error
                        loss_train[i_epoch] += self.loss(y_train) 
                ##### corr_w and b need to be renormalized???
                self.w += corr_w * self.lr
                self.b += corr_b * self.lr
                corr_w, corr_b = 0, 0
            for i_val in range(n_val):
                self.predict(X_val[i_val])
                y_val =  np.copy(Y_val[i_val])
                loss_val[i_epoch] += self.loss(y_val)
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
        #plt.plot(loss_train, color='blue', label="train loss")
        #plt.plot(loss_val, color='red', label="validation loss")
        #plt.title('model loss')
        #plt.ylabel('loss')
        #plt.xlabel('epoch')
        #plt.legend(['Train', 'Validation'], loc='best')
        #plt.savefig("/home/lb_linux/NeuralNetworksFromScratch/train_NNFS.pdf")



"""
    def forward(self, X):
        sum = np.zeros(len(self.layers)+1)       # number of total ind up to each layer
        prod = np.zeros(len(self.layers))        # number of total weights up to each layer
        for i in range(len(self.layers)):
            sum[i+1] = self.layers[i] + sum[i]
            if (i>0):
                prod[i] = self.layers[i-1]*self.layers[i] + prod[i-1]
        sum = sum.astype(int)
        prod = prod.astype(int)
        L = np.zeros(sum[len(self.layers)-1])
        y = np.empty(self.layers[len(self.layers)-1])
        L[0: self.layers[0]] = X
        for i in range(1, len(self.layers)):
            for j in range(self.layers[i]):
                if(i==(len(self.layers)-1)):
                    y[j] = np.sum(L[sum[i-1]:sum[i]] * self.w[prod[i-1]+j*self.layers[i-1]:prod[i-1]+(j+1)*self.layers[i-1]]) + self.b[sum[i]+j]
                else:
                    L[sum[i]+j] = np.sum(L[sum[i-1]:sum[i]] * self.w[prod[i-1]+j*self.layers[i-1]:prod[i-1]+(j+1)*self.layers[i-1]]) + self.b[sum[i]+j]
        self.nodes = L
        self.y_pred = y
        return

"""

