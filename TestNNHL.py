import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

import NNHL


test_1 = False                          # test of weights
test_2 = True                         # test on dataset make_moons



##################################################
# plot keras make_moons
##################################################

if (test_1==True):
    X_train = np.array([ [0, 1, 0],
                [0, 0, 1], 
                [1, 0 ,0], 
                [1, 1 ,0], 
                [1, 1, 1], 
                [0, 1, 1], 
                [0, 1, 0] ])
    Y_train = np.array([1, 0, 0, 1, 1, 0, 1])

    layers = [3, 2, 1]
    model_s = NNHL.NeuralNetwork(layers=layers, seed=7)
    model_s.activation = ['sigmoid', 'sigmoid']
    model_s.w = np.ones(len(model_s.w))
    model_s.forward(X_train[[3]])
    s_pred = model_s.n[-1]

    model_s.fit(X_train=X_train[[3]], Y_train=Y_train[[3]], X_val=X_train[[3]], Y_val=Y_train[[3]], epochs=1, batch_size=1, learn_rate=0.01, shuffle=False)

    model_k = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape = (3,), kernel_initializer=tf.constant_initializer(1.)),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.constant_initializer(1.))])
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model_k.compile(optimizer=opt, loss='mse')
    k_pred = model_k.predict(X_train[[3]])
    model_k.fit(X_train[[3]], Y_train[[3]], epochs=1, batch_size=1)

    for i in range(len(layers)-1):
        w_s = model_s.weight([i+1])
        b_s = model_s.bias([i+1])
        w_k, b_k = model_k.layers[i].get_weights()
        for j in range(layers[i+1]):
            print("\nLayer: {}   node: {}".format(i+1, j+1))
            print("NNFS      weights:{}   bias: {}".format(w_s[j], b_s[j]))
            print("KERAS     weights:{}   bias: {}".format(w_k.transpose()[j], b_k[j]))
    print("\nOutput: \nNNFS   {} \nKERAS  {}".format(s_pred, k_pred[0, 0]))




##################################################
# plots make_moons
##################################################

if (test_2==True):

    seeds = [7, 8, 9]

    for seed_all in seeds:
        np.random.seed(seed_all) 
        tf.random.set_seed(seed_all) 


        # create training samples
        X_data, Y_data = datasets.make_moons(1000, noise = 0.1)
        X_train, X_val, Y_train, Y_val=train_test_split(X_data, Y_data, test_size=0.25, random_state=seed_all)
        fig, axs = plt.subplots(2, figsize=(8.27, 11.69))
        axs[0].scatter(X_data[:,0], X_data[:,1], c=Y_data, cmap=plt.cm.Spectral)
        axs[1].scatter(X_data[:,0], X_data[:,1], c=Y_data, cmap=plt.cm.Spectral)

        # define the neural network NNFS 
        layers = [2, 4, 4, 4, 1]
        model_s = NNHL.NeuralNetwork(layers=layers, seed=7)
        model_s.activation = ['relu', 'sigmoid']
        w_s_i = np.array(model_s.w)

        # training NNFS
        model_s.fit(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, epochs=100, batch_size=10, learn_rate=0.01, patience=10, min_improvement=0)
        
        # define the neural network Keras
        model_k = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='relu', input_shape = (2,)),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])

        # training Keras
        #opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        model_k.compile(optimizer='adam', loss='mse')
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model_k.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=100, callbacks=[callback]) 
        
        # plot decision region
        x_min = X_train[:, 0].min() - 0.5                                        # x and y limit for the grid
        x_max = X_train[:, 0].max() + 0.5 
        y_min = X_train[:, 1].min() - 0.5
        y_max = X_train[:, 1].max() + 0.5
        res = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))   # create a grid that covers the samples
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        Y_pred_s = np.empty(len(X_grid)) 
        for i in range(len(X_grid)):
            Y_pred_s[i] = model_s.predict(X_grid[i])                                 # predicted outputs over the grid
        Y_pred_k = np.empty(len(X_grid)) 
        Y_pred_k = model_k.predict(X_grid)
        Y_grid_s = np.reshape(Y_pred_s, xx.shape)
        Y_grid_k = np.reshape(Y_pred_k, xx.shape)
        axs[0].contourf(xx, yy, Y_grid_s, levels=[0, 0.5, 1], alpha = 0.5)
        axs[0].set_title("Decision region plot of NNFS with layers: {}".format(layers))
        axs[1].contourf(xx, yy, Y_grid_k, levels=[0, 0.5, 1], alpha = 0.5)
        axs[1].set_title("Decision region plot of Keras with layers: {}".format(layers))
        #fig.show()
        fig.savefig("/home/lb_linux/NeuralNetworksFromScratch/PlotMoons_layers{}x{}x{}x{}x{}_seed{}.pdf".format(layers[0], layers[1], layers[2], layers[3], layers[4], seed_all))

        for i in range(len(layers)-1):
            w_s = model_s.weight([i+1])
            b_s = model_s.bias([i+1])
            w_k, b_k = model_k.layers[i].get_weights()
            for j in range(layers[i+1]):
                print("\nLayer: {}   node: {}".format(i+1, j+1))
                print("NNFS      weights:{}   bias: {}".format(w_s[j], b_s[j]))
                print("KERAS     weights:{}   bias: {}".format(w_k.transpose()[j], b_k[j]))

