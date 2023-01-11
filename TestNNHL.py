import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import datasets
from sklearn.model_selection import train_test_split
import csv

import NNHL


test_1 = False                          # test of back propagation method
test_2 = False                          # test on dataset make_moons
test_3 = True                          # test on dataset wine_quality


path = "/home/lb_linux/NeuralNetworksFromScratch"





##################################################
# test 1: back-propagation NNFS vs keras
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
# test 2: make_moons classification problem
##################################################

if (test_2==True):

    seeds = [7]

    for seed_all in seeds:
        np.random.seed(seed_all) 
        tf.random.set_seed(seed_all) 


        # create training samples
        X_data, Y_data = datasets.make_moons(1000, noise = 0.2)
        X_train, X_val, Y_train, Y_val=train_test_split(X_data, Y_data, test_size=0.25, random_state=seed_all)

        # define the neural network NNFS 
        layers = [2, 2, 1]
        model_s = NNHL.NeuralNetwork(layers=layers, seed=7)
        model_s.activation = ['relu', 'sigmoid']

        # training NNFS
        model_s.fit(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, epochs=1000, batch_size=100, learn_rate=0.01, patience=10, min_improvement=0)
        
        # define the neural network Keras
        model_k = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation='relu', input_shape = (2,)),
            tf.keras.layers.Dense(1, activation='sigmoid')])

        # training Keras
        #opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        model_k.compile(optimizer=opt, loss='mse')
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model_k.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=3000, batch_size=100, callbacks=[callback]) 

        save_layers = "{}".format(layers[0])
        for i in range(1, len(layers)):
            save_layers += "x{}".format(layers[i])

        with PdfPages("{}/PlotMoons_layers{}_seed{}.pdf".format(path, save_layers, seed_all)) as pdf: 

            # plot training and validation
            fig, axs = plt.subplots(2, figsize=(8.27, 11.69))             
            axs[0].plot(model_s.loss_train, color='blue', label="training NNFS")
            axs[0].plot(model_s.loss_val, color='red', label="validation NNFS")
            axs[0].set_title('model loss NNFS')
            axs[0].set(xlabel="epoch", ylabel="loss")
            axs[0].set_yscale('log')
            axs[0].legend(['Train', 'Validation'], loc='best')
            axs[1].plot(history.history['loss'], color='blue', label="training Keras")
            axs[1].plot(history.history['val_loss'], color='red', label="validation Keras")
            axs[1].set_title('model loss Keras')
            axs[1].set(xlabel="epoch", ylabel="loss")
            axs[1].set_yscale('log')
            axs[1].legend(['Train', 'Validation'], loc='best')
            pdf.savefig(fig)
            
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
            fig, axs = plt.subplots(2, figsize=(8.27, 11.69))             
            axs[0].scatter(X_data[:,0], X_data[:,1], c=Y_data, cmap=plt.cm.Spectral)
            axs[0].contourf(xx, yy, Y_grid_s, levels=[0, 0.5, 1], alpha = 0.5)
            axs[0].set_title("Decision region plot of NNFS with layers: {}".format(layers))
            axs[1].scatter(X_data[:,0], X_data[:,1], c=Y_data, cmap=plt.cm.Spectral)
            axs[1].contourf(xx, yy, Y_grid_k, levels=[0, 0.5, 1], alpha = 0.5)
            axs[1].set_title("Decision region plot of Keras with layers: {}".format(layers))
            pdf.savefig(fig)


        # print weights NNFS vs Keras
        if False:
            for i in range(len(layers)-1):
                w_s = model_s.weight([i+1])
                b_s = model_s.bias([i+1])
                w_k, b_k = model_k.layers[i].get_weights()
                for j in range(layers[i+1]):
                    print("\nLayer: {}   node: {}".format(i+1, j+1))
                    print("NNFS      weights:{}   bias: {}".format(w_s[j], b_s[j]))
                    print("KERAS     weights:{}   bias: {}".format(w_k.transpose()[j], b_k[j]))





##################################################
#  test 3: weights prediction of pp->tt~ scattering
##################################################

if (test_3==True):

    seed_all = 7
    layers = [5, 64, 64, 1]

    # reading dataset
    data = np.empty((16000, 7)) 
    with open("{}/info_wgt_events.txt".format(path), 'r') as infof:
        print("Start readind events")
        event = 0
        # data in info: px_t, py_t, pz_t, E_t, pz_tbar, E_tbar, wgt 
        for line in infof.readlines():
            data[event, :] = [float(i) for i in line.split()]
            event +=1


    # Physics functions
    def energy_cm(X):
        res = np.sqrt((X[:, 3]+X[:, 5])**2 - (X[:, 2]+X[:, 4])**2)
        return res

    def beta(X):                                     # beta of top-antitop in the lab frame
        res = (X[:, 2]+X[:, 4]) / (X[:, 3]+X[:, 5])
        return res

    def rapidity(X):                                 # rapidity of top in the lab frame
        res = 0.5 * np.log((X[:, 3]+X[:, 2]) / (X[:, 3]-X[:, 2]))
        return res


    # input normalization
    E_cm_pro = 13000                                 # energy of cm of protons
    E_cm_int = energy_cm(data[:, :-1])
    beta_int = beta(data[:, :-1])
    X_data = np.empty(shape=data[:, :-2].shape)
    X_data[:, 0] = E_cm_int / E_cm_pro 
    X_data[:, 1] = rapidity(data[:, :-1])
    X_data[:, 2] = data[:, 0] / E_cm_int
    X_data[:, 3] = data[:, 1] / E_cm_int 
    X_data[:, 4] = (-(beta_int/np.sqrt(1-beta_int**2))*data[:, 3] + (1/np.sqrt(1-beta_int**2))*data[:, 2]) / E_cm_int
    X_train, X_val = X_data[:-4000, :], X_data[-4000:, :]


    # output inizialization
    Y_train, Y_val = data[:-4000, -1], data[-4000:, -1]


    # define the model
    model_s = NNHL.NeuralNetwork(layers=layers, seed=7)
    model_s.activation = ['relu', 'linear']

    tf.random.set_seed(seed_all) 
    model_k = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape = (5, )),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
        ])


    # training and test
    model_s.fit(X_train=X_train, Y_train=np.abs(np.log(Y_train)), X_val=X_val, Y_val=np.abs(np.log(Y_val)), epochs=100, batch_size=10, learn_rate=0.01, patience=30, min_improvement=0)

    model_k.compile(optimizer='adam', loss='mse') 
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    history = model_k.fit(X_train, np.abs(np.log(Y_train)), validation_data=(X_val, np.abs(np.log(Y_val))), batch_size=1000, epochs=100, callbacks=[callback])         # predict the abs(log(w))


    # prediction
    Y_pred_s = np.empty_like(Y_val)
    for i in range(len(X_val)):
        Y_pred_s[i] = model_s.predict(X_val[i])                                 # predicted outputs over the grid
    Y_pred_s = np.e**(-Y_pred_s)                         # model predict -log(w)

    Y_pred_k = model_k.predict(X_val)
    Y_pred_k = np.reshape(Y_pred_k, len(Y_pred_k))
    Y_pred_k = np.e**(-Y_pred_k)                         # model predict -log(w)


    # plot results
    save_layers = "{}".format(layers[0])
    for i in range(1, len(layers)):
        save_layers += "x{}".format(layers[i])

    with PdfPages("{}/PlotWeights_layers{}_seed{}.pdf".format(path, save_layers, seed_all)) as pdf: 

        # plot training and validation
        fig, axs = plt.subplots(2, figsize=(8.27, 11.69))             
        axs[0].plot(model_s.loss_train, color='blue', label="training NNFS")
        axs[0].plot(model_s.loss_val, color='red', label="validation NNFS")
        axs[0].set_title('model loss NNFS')
        axs[0].set(xlabel="epoch", ylabel="loss")
        axs[0].set_yscale('log')
        axs[0].legend(['Train', 'Validation'], loc='best')
        axs[1].plot(history.history['loss'], color='blue', label="training Keras")
        axs[1].plot(history.history['val_loss'], color='red', label="validation Keras")
        axs[1].set_title('model loss Keras')
        axs[1].set(xlabel="epoch", ylabel="loss")
        axs[1].set_yscale('log')
        axs[1].legend(['Train', 'Validation'], loc='best')
        pdf.savefig(fig)

        # plot histograms of predictions
        fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 
        bins_Y = np.logspace(np.log10(0.0001), np.log10(0.003), 50)
        axs[0].set(xlabel="Y", ylabel="dN/d(Y)")
        axs[0].hist(x=Y_val, bins=bins_Y, label="Y_true", color='purple', histtype='step', lw=3, alpha=0.5)
        axs[0].hist(x=Y_pred_s, bins=bins_Y, label="Y_pred_NNFS", color='green', histtype='step', lw=3, alpha=0.5)
        axs[0].hist(x=Y_pred_k, bins=bins_Y, label="Y_pred_Keras", color='orange', histtype='step', lw=3, alpha=0.5)
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].legend(loc='best')

        bins_YY = np.logspace(np.log10(10**(-2)), np.log10(10**(2)), 50)
        axs[1].set(xlabel="Y_true", ylabel="Y_true/Y_pred_NNFS")
        h2 = axs[1].hist2d(Y_val, Y_val/Y_pred_s, bins=[bins_Y, bins_YY])
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        #axs[1].set_title(label="NNFS predictions")
        plt.colorbar(h2[3], ax=axs[1]) 

        axs[2].set(xlabel="Y_true", ylabel="Y_true/Y_pred_Keras")
        h3 = axs[2].hist2d(Y_val, Y_val/Y_pred_k, bins=[bins_Y, bins_YY])
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        #axs[2].set_title(label="Keras predictions")
        plt.colorbar(h3[3], ax=axs[2]) 

        pdf.savefig(fig)
