# Neural Network Practice
import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if deriv == True:
        return x*(1-x)

    return 1 / (1 + np.exp(-x))

# Input dataset
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# Output dataset
Y = np.array([[0, 1, 1, 0]]).T

# Seed random numbers to make calculation deterministic
np.random.seed(1)

# Initialise weights randomly with mean 0
syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

for j in xrange(60000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # By how much did we miss?
    l2_error = Y - l2

    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    # In what direction is the target value? Were we really sure? If so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # How much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # Multiply how much we missed by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # Update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


print "Output:", l2
