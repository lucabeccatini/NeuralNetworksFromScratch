import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import NNHL


# define the neural network and the layers number and size
# default values are: epochs = 10000, lr = 0.1, seed = 7
nn = NNHL.NeuralNetwork(layers_number = 2, layers_size = 2, epochs = 1000, seed = 7)

# create training samples
X_train, y_data = datasets.make_moons(500, noise = 0.2)
plt.figure(figsize = (20, 14))
plt.scatter(X_train[:,0], X_train[:,1], c = y_data, cmap = plt.cm.Spectral)
Y_train = np.zeros([len(X_train), 2])
for i in range(len(X_train)):
    Y_train[i][y_data[i]] = 1    

# train the nn 
nn.training(X_train, Y_train)

# plot decision region
nn.plot_decision_regions(X_train, y_data) 
