import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in range(num_classes):
      if j == y[i]: # Doing nothing because doesn't make sense to compute this case. Is always zero.
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: # else if margin <= 0: no loss will be accumulated, dW is also zero
        count +=1
        loss += margin
        dW[:,j] += X[i].T  # 2nd case: (j!= y_i), see http://cs231n.github.io/optimization-1/#analytic

    dW[:,y[i]] += -count * X[i].T # 1st case: (j== y_i), see http://cs231n.github.io/optimization-1/#analytic

  # Aside: dW[:,j] rather than dW[i,j] is correct. See the equation. Even in
  # this non-vectorized form it is $w_j^T * x_i$ and not sth like $w_{i,j}^T * x_{i,j}$

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train # Must be averaged, otherwise dW would depend on num_train
  dW += reg * W
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)  #scores.shape = (500,10)
  # arange() and y get you the 2D-index of scores with correct s_{i,j}
  correct_class_score = scores[np.arange(num_train), y]
  # each col of matrix scores is substracted by correct class scores (and then delta added)
  margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + delta)
  # again this np.arange() trick gives indices with correct scores.
  # Set all margins to zero for j == y_i
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Semi-vectorized version. It's not any faster than the loop version.
  # num_classes = W.shape[1]
  # incorrect_counts = np.sum(margins > 0, axis=1)
  # for k in xrange(num_classes):
  #   # use the indices of the margin array as a mask.
  #   wj = np.sum(X[margins[:, k] > 0], axis=0)
  #   wy = np.sum(-incorrect_counts[y == k][:, np.newaxis] * X[y == k], axis=0)
  #   dW[:, k] = wj + wy

  # Fully vectorized version. Roughly 10x faster.
  X_mask = np.zeros(margins.shape)
  # column maps to class, row maps to sample; a value v in X_mask[i, j]
  # adds a row sample i to column class j with multiple of v
  X_mask[margins > 0] = 1
  # for each sample, find the total number of classes where margin > 0
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  dW /= num_train  # average out weights


  dW += reg * W  # regularize the weights
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
