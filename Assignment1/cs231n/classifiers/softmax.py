import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.                           (3073, 10)
  - X: A numpy array of shape (N, D) containing a minibatch of data.               (500, 3073)
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means      (500)
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # (3073, 10)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]  # 10
  num_train = X.shape[0]    # 500
  num_dim = W.shape[0]      # 3073
  
  # print("num_classes=", num_classes, "num_train=", num_train, "num_dim", num_dim)
  
  # loss
  for i in range(num_train):
    scores = X[i].dot(W) # (500, 3073)
    expscores = np.copy(scores) # (500, 3073)
    der = 0
    
    # exp
    for j in range(num_classes):
      expscores[j] = np.exp(expscores[j])
      der += expscores[j]
    # norm
    for j in range(num_classes):
      expscores[j] /= der
    loss += -1 * np.log(expscores[y[i]])
    
    expscores[y[i]] -= 1
    for j in range(num_dim):
      for k in range(num_classes):
        dW[j, k] += X[i, j] * expscores[k] 
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg * W
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
  num_classes = W.shape[1]  # 10
  num_train = X.shape[0]    # 500
  num_dim = W.shape[0]      # 3073
  
  score = np.dot(X, W) # (500, 3073)
  expscores = np.exp(score)
  expscores /= np.sum(expscores, axis=1, keepdims=True) # (500, 3073)
  loss -= np.sum(np.log(expscores[np.arange(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)

  expscores[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, expscores)  # (3073, 10)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

