import numpy as np
from random import shuffle
from copy import copy

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # unnormalized log prob of class, s = Wx

    X_train = X.shape[0]
    
    # compute cross entropy loss
    for i in range(X_train):
      s = X[i, :].dot(W)
      # normalize to avoid instability
      s -= np.max(s)

      # softmax = exp(fm) / sum(exp(fm))
      # cross-entropy loss = -log(softmax)
      s_true = s[y[i]]
      s_sum = np.sum(np.exp(s))
      sm = np.exp(s_true) / s_sum
      loss += -np.log(sm)

      for j in range(W.shape[1]):
        if j == y[i]: # if true label
          # dW = (exp(fm) / sum(exp(fm)) - 1) xi
          dW[:, j] += (np.exp(s[j]) / s_sum - 1) * X[i, :]
        else:
          # dW = (exp(fm) / sum(exp(fm))) xi
          dW[:, j] += np.exp(s[j]) / s_sum * X[i, :]

    loss /= X_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= X_train
    dW += reg * W
    ###############################################
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    X_sample = X.shape[0]

    s = X.dot(W)
    s -= np.max(s, axis=1)[:, np.newaxis]

    # cross-entropy loss = -log(exp(fm) / sum(exp(fm))) | fm -> t1

    t1 = -s[np.arange(X_sample), y]
    loss = -np.sum(np.log(np.exp(t1) / np.sum(np.exp(s), axis=1)))

    loss /= X_sample
    loss += 0.5 * reg * np.sum(W * W)

    coef = np.zeros_like(s)
    coef[np.arange(X_sample), y] = 1
    dW = X.T.dot(np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True) - coef)

    dW /= X_sample
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

