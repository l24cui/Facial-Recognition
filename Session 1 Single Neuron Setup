import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# perceptron only valid for 2-dimensional data sets
def perceptron(X, w):
    """
    X is the vector representing data (x,y).
    w is the weight vector and the bias (bias is the last element in w).
    """
    ydiff = X[0]*w[0] + X[1]*w[1] + w[2]
    return ydiff >= 0
    
# Training algorithm for the perceptron to do learning
def trainPerceptron(dataset, labels, perceptron):
    """
    dataset: a numpy array with shape (number of inputs, dimension of inputs)
    labels: a numpy array with shape (number of inputs, 1); the labels are either 0 or 1
    perceptron: your perceptron function; it must consume the input vector, and a vector of weights
    
    The function returns a numpy array with the weights. The last weight is the bias.
    """         
    weights = np.random.normal(0,1,[3])
    for _ in range(1000): # loop to enhance the accuracy
        for i in range(dataset.shape[0]):
            ypred = perceptron(dataset[i][:],weights[:])
            yerr = labels[i,0] - ypred
            if (yerr):
                weights[:-1] = weights[:-1] + yerr * dataset[i,:] # update vector of weight into weights array
                weights[-1] = weights[-1] + yerr ## update the bias
    return weights # returns the last vector of the weight vector (including bias)

# Problem 1: Vertical Separation
def problem1(perceptron=None, weights=None, training=None):
    pop1 = np.random.normal([-3, 0], [ 0.7, 0.7], size=[100,2])
    pop2 = np.random.normal([ 3, 0], [ 0.7, 0.7], size=[100,2])
    plt.scatter(pop1[:,0], pop1[:,1], color='blue')
    plt.scatter(pop2[:,0], pop2[:,1], color='green')
    if not(perceptron is None):
        if weights is None and training is None:
            print('One of weights or training must be set if a perceptron is passed')
        elif not(weights is None):
            pass
        elif not(training is None):
            dataset = np.concatenate([pop1,pop2], axis=0)
            labels = np.concatenate([np.zeros(100),np.ones(100)]).reshape(200,1)
            weights = training(dataset, labels, perceptron)
            print('Your training algorithm got the following weights: {}'.format(weights))
        
        size = 500
        X,Y = np.meshgrid(np.linspace(-10,10,size),np.linspace(-10,10,size))
        x = np.stack([X,Y], axis=-1)
        Z = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                Z[i,j] = perceptron(x[i,j,:], weights)
        plt.contour(X,Y,Z, [0])
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])
    
# Test the basic perceptron
problem1(perceptron, weights = [1,0,0])
    
# Test the training perceptron
problem1(perceptron, training = trainPerceptron)

# Problem 2: Horizontal Separation
def problem2(perceptron=None, weights=None, training=None):
    pop1 = np.random.normal([-1.5, 1.5], [ 0.7, 1.7], size=[100,2])
    pop2 = np.random.normal([ 0, -5], [ 0.4, 0.7], size=[100,2])
    plt.scatter(pop1[:,0], pop1[:,1], color='blue')
    plt.scatter(pop2[:,0], pop2[:,1], color='green')
    if not(perceptron is None):
        if weights is None and training is None:
            print('One of weights or training must be set if a perceptron is passed')
        elif not(weights is None):
            pass
        elif not(training is None):
            dataset = np.concatenate([pop1,pop2], axis=0)
            labels = np.concatenate([np.zeros(100),np.ones(100)]).reshape(200,1)
            weights = training(dataset, labels, perceptron)
            print('Your training algorithm got the following weights: {}'.format(weights))

        size = 500
        X,Y = np.meshgrid(np.linspace(-10,10,size),np.linspace(-10,10,size))
        x = np.stack([X,Y], axis=-1)
        Z = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                Z[i,j] = perceptron(x[i,j,:], weights)
        plt.contour(X,Y,Z, [0])
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])
    
# Test the basic perceptron
problem2(perceptron, weights=[0,1,3])

# Test the training perceptron
problem2(perceptron, training = trainPerceptron)

# Problem 3: Diagonal Separation
def problem3(perceptron=None, weights=None, training=None):
    pop1 = np.zeros([200,2])
    for i,j in enumerate(((0,0), (1.5,1.5), (-1.5,-1.5), (-3,-3.5))):
        pop1[50*i:50*(i+1),:] = np.random.normal(j, [ 0.7, 0.7], size=[50,2])
    pop2 = np.random.normal([ 1, -5], [ 0.6, 0.7], size=[100,2])
    plt.scatter(pop1[:,0], pop1[:,1], color='blue')
    plt.scatter(pop2[:,0], pop2[:,1], color='green')
    if not(perceptron is None):
        if weights is None and training is None:
            print('One of weights or training must be set if a perceptron is passed')
        elif not(weights is None):
            pass
        elif not(training is None):
            dataset = np.concatenate([pop1,pop2], axis=0)
            labels = np.concatenate([np.zeros(200),np.ones(100)]).reshape(300,1)
            weights = training(dataset, labels, perceptron)
            print('Your training algorithm got the following weights: {}'.format(weights))
        
        size=500
        X,Y = np.meshgrid(np.linspace(-10,10,size),np.linspace(-10,10,size))
        x = np.stack([X,Y], axis=-1)
        Z = np.zeros([size,size])
        for i in range(size):
            for j in range(size):
                Z[i,j] = perceptron(x[i,j,:], weights)
        plt.contour(X,Y,Z, [0])
    plt.xlim([-7, 7])
    plt.ylim([-7, 5])

# Test the basic perceptron
problem3(perceptron, weights=[1,-1,-3])

# Test the training perceptron
problem3(perceptron, training = trainPerceptron)

# Problem 4: Radical Symmetry
def problem4(perceptron=None, weights=None, training=None):
    r = np.random.normal(5, 0.5, size=(500,1))
    theta = np.random.uniform(0, 2*np.pi, size=(500,1))
    pop1 = np.concatenate([r*np.cos(theta), r*np.sin(theta)], axis=-1)
    pop2 = np.random.normal([ 0, 0], [ 0.5, 0.5], size=[100,2])
    plt.scatter(pop1[:,0], pop1[:,1], color='blue')
    plt.scatter(pop2[:,0], pop2[:,1], color='green')
    if not(perceptron is None):
        if weights is None and training is None:
            print('One of weights or training must be set if a perceptron is passed')
        elif not(weights is None):
            pass
        elif not(training is None):
            dataset = np.concatenate([pop1,pop2], axis=0)
            labels = np.concatenate([np.zeros(500),np.ones(100)]).reshape(600,1)
            weights = training(dataset, labels, perceptron)
            print('Your training algorithm got the following weights: {}'.format(weights))

        size = 500
        X,Y = np.meshgrid(np.linspace(-10,10,size),np.linspace(-10,10,size))
        x = np.stack([X,Y], axis=-1)
        Z = np.zeros([size,size])
        for i in range(size):
            for j in range(size):
                Z[i,j] = perceptron(x[i,j,:], weights)
        plt.contour(X,Y,Z, [0])
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])
    
# define a new perceptron (using the existed one) for a radical symmetry model
def percRad(X, w):
    """
    X is the vector representing data (x,y).
    w is the weight vector and the bias (bias is the last element in w).
    Here is where your code should go.
    Remember the bias term.
    """
    ## using polar coordinates
    r = (X[0]**2 + X[1]**2)**0.5
    Theta = np.arctan2(X[1],X[0])
    X = np.stack([r,Theta],axis=-1)
    return perceptron(X,w)
    
# Test the algorithm on radical model
problem4(percRad, training=trainPerceptron)

################################################################################
# The tricky part: More than 2 dimensions
################################################################################

# Dealing with data - a subset of the MNIST data set: only pictures (28px by 28px)
# of zeros and ones - as following:
dataset = np.load('dataset_01.npy')
zero = dataset[0,:,:]
one  = dataset[6000,:,:]

plt.subplot(121)
plt.imshow(zero, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Zero')
plt.subplot(122)
plt.imshow(one, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('One');

# Problem 5: More than 2 dimensions
def problem5(perceptron=None):
    dataset = np.load('dataset_01.npy')/255
    labels = np.load('labels_01.npy')
    
    mean = np.mean(dataset, axis=0)
    std  = np.std(dataset, axis=0)
    whiteData = np.nan_to_num((dataset - mean)/std).reshape(12665 ,784)
    cov = whiteData.T @ whiteData
    U, _, _ = np.linalg.svd(cov)
    
    if perceptron!=None:
        weights = trainPerceptron(whiteData, labels, perceptron)

    x =  whiteData @ U[:,:2]
    plt.scatter(x[:5923,0], x[:5923,1], color='blue', s=2)
    plt.scatter(x[5923:,0], x[5923:,1], color='green', s=2)
    
    if perceptron!=None:        
        size = 500
        X,Y = np.meshgrid(np.linspace(-40,25,size),np.linspace(-40,50,size))
        x = np.stack([X,Y], axis=-1)
        Z = np.zeros([size,size])
        for i in range(size):
            for j in range(size):
                Z[i,j] = perceptron(U[:,:2] @ x[i,j,:], weights)
        plt.contour(X,Y,Z, [0])
    plt.xlim([-30,15])
    plt.ylim([-30,40])

# A new definition to perceptron in order to accommodate larger-dimensional model
def biggerPerceptron(X,w):
    """
    X is the vector representing data vector.
    w is the weight vector and the bias (bias is the last element in w).
    """
    ydiff = w[:-1] @ X + w[-1] # now calculate the vectors rather than their entries
    return ydiff >= 0
    
# A new definition to training perceptron in order to accommodate larger-dimensional model
def trainPerceptron(dataset, labels, perceptron):
    """
    dataset: a numpy array with shape (number of inputs, dimension of inputs)
    labels: a numpy array with shape (number of inputs, 1); the labels are either 0 or 1
    perceptron: your perceptron function; it must consume the input vector, and a vector of weights
    
    The function returns a numpy array with the weights. The last weight is the bias.
    """   
    # it's the same with the previous one (except the dimension of weight vectors in weights)
    # since I did that in vector calculation way rather than vector elements calculation way
    weights = np.random.normal(0,1,[784+1]) # this is only line to be changed from previous definition
    for _ in range(1000): # loop to enhance the accuracy
        for i in range(dataset.shape[0]):
            ypred = perceptron(dataset[i][:],weights[:])
            yerr = labels[i,0] - ypred
            if (yerr):
                weights[:-1] = weights[:-1] + yerr * dataset[i,:] # update vector of weight into weights array
                weights[-1] = weights[-1] + yerr # update the bias
    return weights # returns the last vector of the weight vector (including bias)
    
# Test the algorithm on larger-dimensional model
problem5(biggerPerceptron)
