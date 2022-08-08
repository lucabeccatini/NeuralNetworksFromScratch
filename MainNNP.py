import numpy as np
import NNP


# define the perceptron and the number of input variables
perp = NNP.Perceptron(3)

# load training samples
X_train = np.array([ [0, 1, 0],
            [0, 0, 1], 
            [1, 0 ,0], 
            [1, 1 ,0], 
            [1, 1, 1], 
            [0, 1, 1], 
            [0, 1, 0] ])
Y_train = np.array([1, 0, 0, 1, 1, 0, 1])

# train the perceptron
perp.training(X_train, Y_train)

# predict 
x = np.array([1, 0, 0])
perp.predict(x)