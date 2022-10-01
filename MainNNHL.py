import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import NNHL


# define the neural network and the layers number and size
# default values are: epochs = 10000, lr = 0.1, seed = 7
model = NNHL.NeuralNetwork(layers_number = 2, layers_size = 2, seed = 7)

# create training samples
X_data, Y_data = datasets.make_moons(1000, noise = 0.2)
X_train, X_val, Y_train_labels, Y_val_labels = train_test_split(X_data, Y_data, test_size=0.333, random_state=7)
plt.figure(figsize = (20, 14))
plt.scatter(X_val[:,0], X_val[:,1], c = Y_val_labels, cmap = plt.cm.Spectral)
Y_train = np.zeros([len(X_train), 2])
for i in range(len(X_train)):
    Y_train[i][Y_train_labels[i]] = 1
Y_val = np.zeros([len(X_val), 2])    
for j in range(len(X_val)):
    Y_val[j][Y_val_labels[j]] = 1    

# train the nn 
model.learn(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, epochs = 1000, batch_size=10, patience=10, min_improvement=0)

# plot decision region
model.plot_decision_regions(X_val, Y_val_labels) 
