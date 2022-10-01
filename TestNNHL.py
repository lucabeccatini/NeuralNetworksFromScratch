import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets



# test that wheight and inout multiply correctly (w for 3 layers not correct)

##################################################
# test NNHL with keras comparing the results
##################################################

# create training samples
np.random.seed(7) 
X_train, y_data = datasets.make_moons(500, noise = 0.2)
Y_train = np.zeros([len(X_train), 2])
for i in range(len(X_train)):
    Y_train[i][y_data[i]] = 1
plt.figure(figsize = (20, 14)) 
plt.scatter(X_train[:,0], X_train[:,1], c = y_data, cmap = plt.cm.Spectral)

# create the nn
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape = (2,)),
    tf.keras.layers.Dense(2, activation='softmax')])

# loss function and compile the model 
Y_pred = model(X_train[:1]).numpy()
loss_fn = loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

#training
model.fit(X_train, Y_train, epochs=100)

# plot decision region
x_min = X_train[:, 0].min() - 0.5                                        # x and y limit for the grid
x_max = X_train[:, 0].max() + 0.5 
y_min = X_train[:, 1].min() - 0.5
y_max = X_train[:, 1].max() + 0.5
res = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))   # create a grid that covers the samples
X_grid = np.c_[xx.ravel(), yy.ravel()]
Y_grid = np.empty([len(X_grid), 2]) 
Z_pred = np.zeros(len(X_grid)) 
Y_grid = model.predict(X_grid)
for m in range(len(X_grid)):
    #Y_grid[m] = model.predict(X_grid[m])                                 # predicted outputs over the grid
    if m % 10 == 0:
        print("Y[{}] = {}".format(m, Y_grid[m]))
    for n in range(2): 
        Z_pred[m] += (n+1) * (Y_grid[m][n] > 0.5).astype(int)          # needed a single value per point for contourf
    #print(Z_pred[m]) 
Z_pred = np.reshape(Z_pred, xx.shape)
lev = [0, 1, 2, 3]
plt.contourf(xx, yy, Z_pred, levels = lev, alpha = 0.5)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.scatter(X_train[:, 0], X_train[:, 1], c = y_data,  alpha = 0.2)     # .. c=Y_true.reshape(-1),  alpha=0.2)
plt.title("Decision region plot of NN with {} layers and {} layers size".format(2, 2))