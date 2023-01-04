import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import NNHL



# define the neural network and the layers number and size
# default values are: epochs = 100, lr = 0.01, seed = 7
layers = [2, 2, 2]
model_s = NNHL.NeuralNetwork(layers=layers, seed=7)


# create training samples
X_data, Y_data_labels = datasets.make_moons(500, noise = 0.2)
X_train, X_val, Y_train_labels, Y_val_labels=train_test_split(X_data, Y_data_labels, test_size=0.25, random_state=7)
fig, axs = plt.subplots(2, figsize=(8.27, 11.69))
axs[0].scatter(X_data[:,0], X_data[:,1], c=Y_data_labels, cmap=plt.cm.Spectral)
axs[1].scatter(X_data[:,0], X_data[:,1], c=Y_data_labels, cmap=plt.cm.Spectral)
Y_train = np.zeros([len(X_train), 2])
for i in range(len(X_train)):
    Y_train[i, Y_train_labels[i]] = 1
Y_val = np.zeros([len(X_val), 2])    
for i in range(len(X_val)):
    Y_val[i, Y_val_labels[i]] = 1 


# train the nn 
model_s.fit(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, epochs=100, batch_size=10, patience=10, min_improvement=0)


# plot decision region
if (True):
    x_min = X_data[:, 0].min() - 0.5                                        # x and y limit for the grid
    x_max = X_data[:, 0].max() + 0.5 
    y_min = X_data[:, 1].min() - 0.5
    y_max = X_data[:, 1].max() + 0.5
    res = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))   # create a grid that covers the samples
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Y_pred_s = np.empty([len(X_grid), 2]) 
    Z_pred_s = np.zeros(len(X_grid)) 
    for i in range(len(X_grid)):  
        Y_pred_s[i] = model_s.predict(X_grid[i])                                 # predicted outputs over the grid
        if (Y_pred_s[i, 1]>Y_pred_s[i, 0]):
            Z_pred_s[i] = 1
    Z_pred_s = np.reshape(Z_pred_s, xx.shape)
    lev = [-1, 0, 1, 2]
    axs[0].contourf(xx, yy, Z_pred_s, levels=lev, alpha = 0.5)
    #axs[0].set_xlim(xx.min(), xx.max())
    #axs[0].set_ylim(yy.min(), yy.max())
    #axs[0].scatter(X_val[:, 0], X_val[:, 1], c = Y_val_labels,  alpha = 0.2)     # .. c=Y_true.reshape(-1),  alpha=0.2)
    axs[0].set_title("Decision region plot of NNFS with layers: {}".format(model_s.layers)) 
    fig.savefig('/home/lb_linux/NeuralNetworksFromScratch/MoonsNNFS.pdf')    
    #fig.show()
