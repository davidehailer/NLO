import numpy as np

def F(w, X, Y):
    """This function implements the nonconvex objective function of an
    optimization problem used to train a neural network.
    
     INPUT
    =======================================
     w ...... point at which the objective is evaluated (vector, dim.: 79510)
     X ...... datapoints (matrix, datapoints are the row vectors)
     Y ...... lables corresponding to datapoints (matrix)
    
     OUTPUT
    =======================================
     F_val .. value of objective function at w (scalar)
     """

    [numb_samples, inputnodes] = X.shape
    H = h(w, X)
    B = Y.astype(bool)
    log_H = np.log(H[B])
    F_val = -(1/numb_samples) * np.sum(log_H)
    return F_val


def F_gradient(w, X, Y):
    """This function implements the gradient of an objective function of an
    optimization problem used to train a neural network by backpropergation.
    
     INPUT
    =======================================
     w ...... point at which the objective is evaluated (vector, dim.: 79510)
     X ...... datapoints (matrix, datapoints are the row vectors)
     Y ...... lables corresponding to datapoints (matrix)
    
     OUTPUT
    =======================================
     F_grad .. gradient of objective function at w (vector, dim.: 79510)
     """
    
    # forward propagation
    [numb_samples, inputnodes] = X.shape
    numb_input = 784
    numb_hidden = 100
    numb_output = 10
    first_layer  = (numb_input + 1) * numb_hidden
    second_layer = (numb_hidden + 1) * numb_output
    numb_weights = first_layer +  second_layer

    W_1 = w[0:first_layer].reshape((numb_input + 1), numb_hidden)   # (785, 100)
    W_2 = w[first_layer:numb_weights].reshape((numb_hidden + 1), numb_output)   # (101, 10)
    
    O_input = np.concatenate((X,np.ones((X.shape[0],1))), axis=1)   # (n,785) <-- (n,784)
    Z_hidden = O_input @ W_1  # (n,100) = (n,785) @ (785, 100)
    O_hidden = logistic(Z_hidden)  # (n,100)
    O_hidden = np.concatenate((O_hidden,np.ones((numb_samples,1))), axis=1) # (n,101) <-- (n, 100)
    Z_output = O_hidden @ W_2  # (n,10) = (n,101) @ (101, 10)
    O_output = softmax(Z_output)  # (n,10)
    
    
    # backward propagation
    D_output = O_output - Y   # (n,10)
    Q = D_output @ W_2[:-1,:].T   # (n,100) = (n,10) @ (10,100), since for the artificial node we don't need delta
    D_hidden = O_hidden[:,:-1] * (1 - O_hidden[:,:-1]) * Q  # (n,100) 
    
    G_1 = O_input.T @ D_hidden   # (785, 100) = (785, n) @ (n, 100)
    G_2 = O_hidden.T @ D_output   # (101, 10) = (101,n) @ (n, 10)    
    
    g = np.zeros(numb_weights)
    g[:first_layer] = G_1.flatten()
    g[first_layer:numb_weights] = G_2.flatten()
    
    return (1/numb_samples) * g


def h(w, X):
    """This function implements the output of a neural net.
    
     INPUT
    =======================================
     w ...... point at which the objective is evaluated (vector, dim.: 79510)
     X ...... datapoints (matrix, datapoints are the row vectors)
    
     OUTPUT
    =======================================
     O_output .. output of neural net for w and X (matrix, predictions in rows)
     """
     
    # forward propagation
    [numb_samples, inputnodes] = X.shape
    numb_input = 784
    numb_hidden = 100
    numb_output = 10
    first_layer  = (numb_input + 1) * numb_hidden
    second_layer = (numb_hidden + 1) * numb_output
    numb_weights = first_layer +  second_layer

    W_1 = w[0:first_layer].reshape((numb_input + 1), numb_hidden)   # (785, 100)
    W_2 = w[first_layer:numb_weights].reshape((numb_hidden + 1), numb_output)   # (101, 10)
    
    O_input = np.concatenate((X,np.ones((X.shape[0],1))), axis=1)   # (n,785) <-- (n,784)
    Z_hidden = O_input @ W_1  # (n,100) = (n,785) @ (785, 100)
    O_hidden = logistic(Z_hidden)  # (n,100)
    O_hidden = np.concatenate((O_hidden,np.ones((numb_samples,1))), axis=1) # (n,101) <-- (n, 100)
    Z_output = O_hidden @ W_2  # (n,10) = (n,101) @ (101, 10)
    O_output = softmax(Z_output)  # (n,10)
    
    return O_output


def logistic(t):    
    """ Returns value of the logistic function."""
    value = np.zeros_like(t, dtype=np.float)
    idx = t < 0
    exp_t = np.exp(t[idx])
    value[idx] = exp_t / (1 + exp_t)
    value[~idx] = 1 / (1 + np.exp(-t[~idx]))
    return value


def softmax(z):    
    """ Returns value of the softmax function."""
    exp_z = np.exp(z)
    if exp_z.ndim == 2:
        exp_sum = np.sum(exp_z, axis = 1)
    else:
        exp_sum = np.sum(exp_z.reshape(1, -1), axis = 1)
    return exp_z/exp_sum[:, None]

