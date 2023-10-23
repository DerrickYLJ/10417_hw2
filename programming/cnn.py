"""
Fall 2023, 10-417/617
Assignment-2
Programming - CNN
TAs in charge: Jared Mejia, Kaiwen Geng

IMPORTANT:
    DO NOT change any function signatures but feel free to add instance variables and methods to the classes.

October 2023
"""

from re import L
import numpy as np
import copy
import pickle
import math
import im2col_helper  # uncomment this line if you wish to make use of the im2col_helper.pyc file for experiments
import matplotlib.pyplot as plt
import argparse

CLASS_IDS = {
 'cat': 0,
 'dog': 1,
 'car': 2,
 'bus': 3,
 'train': 4,
 'boat': 5,
 'ball': 6,
 'pizza': 7,
 'chair': 8,
 'table': 9
 }

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis = 1).reshape(-1,1)

def random_weight_init(input, output):
    b = np.sqrt(6)/np.sqrt(input+output)
    return np.random.uniform(-b, b, (output, input))

def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.

    Note: You must implement im2col yourself. If you use any functions from im2col_helper, you will lose 50
    points on this assignment.
    '''
    #TODO
    # # print(f"k_heigth: {k_height}; k_width: {k_width}; padding: {padding}; stride: {stride}; X.shape: {X.shape}")
    (N, C, H, W) = X.shape
    H_prime = ((H-k_height+2*padding)//stride)+1
    W_prime = ((W-k_width+2*padding)//stride)+1
    # Start Padding
    padded_X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=(0.0))
    # Create Feature Map
    feature_map = np.zeros((C*k_height*k_width, H_prime*W_prime*N))
    count = 0
    
    for h in range(0, H+2*padding,stride):
        # add one might lead to error
        for w in range(0, W+2*padding, stride):
            for n in range(N):
                if (h+k_height) <= H+2*padding and (w+k_width) <= W+2*padding:
                    patch = padded_X[n, :, h:(h+k_height), w:(w+k_width)].reshape(k_height*k_width*C,)
                    feature_map[:, count] = patch
                    count += 1
    return feature_map



def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape

    Note: You must implement im2col yourself. If you use any functions from im2col_helper, you will lose 50
    points on this assignment.
    '''
    #TODO
    (N, C, H, W) = X_shape
    feature_map = np.zeros((N, C, H+2*padding, W+2*padding))
    # feature_map = np.zeros((N, C, H, W))
    H_prime = ((H-k_height+2*padding)//stride)+1
    W_prime = ((W-k_width+2*padding)//stride)+1
    (K2C, HWN) = grad_X_col.shape
    for j in range(HWN):
        batch_index = j%N
        total_patches = j//N
        h = total_patches//W_prime * stride
        w = total_patches%W_prime * stride
        patch = grad_X_col[:,j].reshape((1,C,k_height,k_width))
        feature_map[batch_index, :, h:h+k_height, w:w+k_width] = feature_map[batch_index, :, h:h+k_height, w:w+k_width] + patch
    # shrink the feature_map to the appropriate dimension
    if padding == 0:
        return feature_map
    else:
        return feature_map[:,:,padding:-padding, padding:-padding]



class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Note: we are not going to be accumulating gradients (where in hw1 we did)
        In each forward and backward pass, the gradients will be replaced.
        Therefore, there is no need to call on zero_grad().
        This is functionally the same as hw1 given that there is a step along the optimizer in each call of forward, backward, step
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)

    def forward(self, x):
        #TODO
        self.p = (x > 0.0)*x
        return self.p

    def backward(self, grad_wrt_out):
        #TODO
        drelu = (self.p > 0.0) 
        return drelu*grad_wrt_out
    

class Dropout(Transform):
    """
    Implement this class. You may use your implementation from HW1
    """

    def __init__(self, p=0.1):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        self.dropout_probility = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial during training
        Scale your output accordingly during testing ???Error??? how does this mask work
        """
        #TODO
        
        def filter_out(i):
            return 0.0 if i < self.dropout_probility else 1.0
        
        def relu(i):
            return 0.0 if i < 0 else i
        
        vectorized_filter_out = np.vectorize(filter_out)
        vectorized_relu = np.vectorize(relu)
        if train:
            xrand = np.random.uniform(0,1,x.shape)
            filter_rand = vectorized_filter_out(xrand)
            relu_x = vectorized_relu(x)
            dropepd_x = relu_x*filter_rand
            self.p = dropepd_x
        else:
            relu_x = vectorized_relu(x)
            dropepd_x = relu_x*(1-self.dropout_probility)
            self.p = dropepd_x
        return dropepd_x

        
    
    def backward(self, grad_wrt_out):
        """
        This method is only called during trianing.
        """
        # TODO
        def relu1(i):
            return 0.0 if i <= 0 else 1.0
        vectorized_relu = np.vectorize(relu1)
        relu_p = vectorized_relu(self.p)
        # print(relu_p.shape)
        # print(grad_wrt_out.shape)
        return relu_p*grad_wrt_out
    


class Flatten(Transform):
    """
    Implement this class
    """
    def forward(self, x):
        """
        x has shape batch_size * num_filter * height * width
        returns Flatten(x)
        """
        (batch_size, num_filter, height, width)= x.shape
        self.shape = x.shape
        return x.reshape(batch_size, num_filter*height*width)
        

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        #TODO
        (batch_size, num_filter, height, width)= self.shape
        return dloss.reshape(self.shape)


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        #TODO
        (channels, height, width) = input_shape
        (num_filters, filter_height, filter_width) = filter_shape
        self.channels = channels
        self.height = height
        self.width = width
        self.num_filters = num_filters
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.weights_shape = (self.num_filters, self.channels, self.filter_height, self.filter_width)
        self.bias = np.zeros((self.num_filters,1))
        b = pow(6/((self.num_filters+self.channels)*self.filter_height*self.filter_width), 0.5)
        self.weights = np.random.uniform(-b, b, self.weights_shape)
        self.Gmw = np.zeros(self.weights_shape)
        self.Gmb = np.zeros((num_filters, 1))
        self.dw = np.array([])
        self.db = np.array([])


    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height', width')
        we recommend you use im2col here
        weights shape (number of filters, number of input channels, filter height, filter width)
        """
        #TODO
       
        (N, C, H, W) = inputs.shape
        self.inputs = inputs
        feature_map = im2col_helper.im2col(inputs, self.filter_height, self.filter_width, pad, stride)
        H_prime = ((H-self.filter_height+2*pad)//stride)+1
        W_prime = ((W-self.filter_width+2*pad)//stride)+1
        # turn to Conv Weights
        conv_weights = np.zeros((self.num_filters, self.filter_height*self.filter_width*self.channels))
        result = np.zeros((N, self.num_filters, H_prime, W_prime))
        for n in range(self.num_filters):
            conv_weights[n, :] = self.weights[n, :, :, :].flatten()
        res_immature = np.dot(conv_weights, feature_map)+self.bias # error? for bias
        rH, rW = res_immature.shape
        # for i in range(rH):
        #     for j in range(rW):
        #         n = j % N
        #         nf = i
        #         h = (j // N) // W_prime
        #         w = (j // N) % W_prime
        #         result[n, nf, h, w] = res_immature[i, j]
        out = res_immature.reshape(self.num_filters, H_prime, W_prime, N)

        self.stride = stride
        self.Hprime = H_prime
        self.Wprime = W_prime
        self.padding = pad
        return out.transpose(3, 0, 1, 2)

        

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        #TODO
        (batch_size, num_filters, output_height, output_width) = dloss.shape
        self.batch_size = batch_size
        padding = self.padding
        stride = self.stride
        dloss_transposed = np.transpose(dloss, (1,2,3,0))
        dloss_transposed_reshaped = dloss_transposed.reshape(((self.num_filters, batch_size*output_height*output_width)))
        # db
        self.db = (np.sum(dloss, axis=(0,2,3))).reshape(num_filters, 1)
        # dw
        feature_map = im2col_helper.im2col(self.inputs, self.filter_height, self.filter_width, padding, stride)
        tmp_dw = np.dot(dloss_transposed_reshaped, feature_map.T)
        self.dw = tmp_dw.reshape(self.weights_shape)
        
        # self.dw = np.zeros(self.weights_shape)
        # padded_X = np.pad(self.inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=(0.0))
        # for n in range(batch_size):
        #     for nf in range(self.num_filters):
        #         for c in range(self.channels):
        #             for hp in range(self.Hprime):
        #                 for wp in range(self.Wprime):
        #                     for h in range(self.filter_height):
        #                         for w in range(self.filter_height):
        #                             self.dw[nf, c, h, w] += dloss[n, nf, hp, wp]*padded_X[n, c, self.stride*hp+h, self.stride*wp+w]
        
        # grad w.r.t inputs
        conv_weights = np.zeros((self.num_filters, self.filter_height*self.filter_width*self.channels))
        for n in range(self.num_filters):
            conv_weights[n, :] = self.weights[n, :, :, :].flatten()
        res_raw = np.dot(conv_weights.T, dloss_transposed_reshaped)
        res_dx = im2col_helper.im2col_bw(res_raw, (batch_size, self.channels, self.height, self.width), self.filter_height, self.filter_width, padding, stride)
        return self.dw, self.db, res_dx


    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Here we divide gradients by batch_size.
        """
       #TODO
        # Gmw_new = np.add(self.Gmw*momentum_coeff, self.dw/self.batch_size)
        # self.weights = self.weights - Gmw_new*learning_rate
        # self.Gmw = self.dw
        # Gmb_new = np.add(self.Gmb*momentum_coeff, self.db/self.batch_size)
        # self.bias = self.bias - Gmb_new*learning_rate
        # self.Gmb = self.db

        self.Gmw = np.add(self.Gmw*momentum_coeff, self.dw/self.batch_size)
        self.weights = self.weights - self.Gmw*learning_rate
        self.Gmb = np.add(self.Gmb*momentum_coeff, self.db/self.batch_size)
        self.bias = self.bias - self.Gmb*learning_rate

        

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        #TODO
        return self.weights, self.bias


class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        #TODO
        self.fitler_shape = filter_shape
        self.stride = stride

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (batch_size, C, H, W)
        """
        #TODO
        
        self.inputs = inputs
        (filter_height, filter_width) = self.fitler_shape
        (batch_size, C, H, W) = inputs.shape
        stride = self.stride
        out_height = ((H-filter_height)//self.stride)+1
        out_width = ((W-filter_width)//self.stride)+1
        # out_height = H // filter_height
        # out_width = W // filter_width
        

        inputs_reshaped = inputs.reshape(batch_size*C, 1, H, W)
        inputs_col = im2col_helper.im2col(inputs_reshaped, filter_height, filter_width, 0, stride)
        max_indices = np.argmax(inputs_col, axis=0)
        res1 = inputs_col[max_indices, range(max_indices.size)]
        res2 = res1.reshape(out_height, out_width, batch_size, C)
        res3 = res2.transpose(2, 3, 0, 1)
        self.inputs_col = inputs_col
        self.max_indices = max_indices
        self.batch_size = batch_size
        self.C = C
        self.H = H
        self.W = W
        self.filter_height = filter_height
        self.filter_width = filter_width

        return res3


        # max_pool = np.zeros((batch_size, C, out_height, out_width))
        # max_indices = np.zeros((batch_size, C, out_height, out_width), dtype=int)
        # count = 0
        # for b in range(batch_size):
        #     for c in range(C):
        #         for i in range(0, H, stride):
        #             for j in range(0, W, stride):
        #                 if i + filter_height <= H and j + filter_width <= W:
        #                     mj = count % (total_per_batch) % out_width
        #                     mi = count % (total_per_batch) // out_width
        #                     max_pool[b, c, mi, mj] = np.max(inputs[b, c, i:i + filter_height, j:j + filter_width])
        #                     count += 1
        # return max_pool

        # for i in range(out_height):
        #     for j in range(out_width):
        #         h_start = i * stride
        #         h_end = h_start + filter_height
        #         w_start = j * stride
        #         w_end = w_start + filter_width
        #         pool_slice = inputs[:, :, h_start:h_end, w_start:w_end]
        #         max_values = np.max(pool_slice, axis=(2, 3))
        #         max_pool[:, :, i, j] = max_values

        

        # output_tensor = np.max(col, axis=0).reshape(inputs.shape[0], -1)
        # self.max_indices = max_indices
        return max_pool

        


    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        dinputs_col = np.zeros(self.inputs_col.shape)
        dloss_flat = dloss.transpose(2, 3, 0, 1).ravel()
        dinputs_col[self.max_indices, range(self.max_indices.size)] = dloss_flat
        inputs_grad = im2col_helper.im2col_bw(dinputs_col, (self.batch_size * self.C, 1, self.H, self.W), self.filter_height, self.filter_width, 0, self.stride)
        inputs_grad = inputs_grad.reshape(self.inputs.shape)



        # inputs = self.inputs
        # (filter_height, filter_width) = self.fitler_shape
        # (batch_size, C, H, W) = inputs.shape
        # (batch_size, C, out_height, out_width) = dloss.shape
        # stride = self.stride
        # inputs_grad = np.zeros(inputs.shape)
        # for b in range(batch_size):
        #     for c in range(C):
        #         for i in range(out_height):
        #             for j in range(out_width):
        #                 window = inputs[b, c, i * filter_height:(i + 1) * filter_height, j * filter_width:(j + 1) * filter_width]
        #                 max_location = np.unravel_index(np.argmax(window), window.shape)
        #                 inputs_grad[b, c, i * filter_height + max_location[0], j * filter_width + max_location[1]] = dloss[b, c, i, j]



        return inputs_grad



class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim), initialized as (outdim,indim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.indim = indim
        self.outdim = outdim
        self.weights_shape = (indim,outdim)
        self.b = np.zeros((self.outdim,1))
        self.w = random_weight_init(self.outdim, self.indim) #??? what is the initialization
        self.Gmw = np.zeros((outdim, indim))
        self.Gmb = np.zeros((outdim, 1))
        self.dw = np.array([])
        self.db = np.array([])


    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        #TODO
        x = inputs
        new_x = np.dot(x, self.w).T + self.b
        self.x = x
        return new_x.T

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        #TODO
        grad_wrt_out = dloss.T
        (outdim,batch) = grad_wrt_out.shape
        self.batch = batch
        self.dw = np.dot(grad_wrt_out, self.x)
        db_sum = np.sum(grad_wrt_out, axis=1)
        db_sum_T = np.reshape(db_sum,(outdim,1))
        self.db = db_sum_T
        A = np.dot(self.w ,grad_wrt_out).T
        return (self.dw.T, self.db, A)


    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        #TODO
        # Gmw_new = np.add(self.Gmw*momentum_coeff, self.dw/self.batch)
        # self.w = self.w - Gmw_new.T*learning_rate
        # self.Gmw = self.dw
        # Gmb_new = np.add(self.Gmb*momentum_coeff, self.db/self.batch)
        # self.b = self.b - Gmb_new*learning_rate
        # self.Gmb = self.db
        self.Gmw = np.add(self.Gmw*momentum_coeff, self.dw/self.batch)
        self.w = self.w - self.Gmw.T*learning_rate
        self.Gmb = np.add(self.Gmb*momentum_coeff, self.db/self.batch)
        self.b = self.b - self.Gmb*learning_rate

    def get_wb_fc(self):
        """
        Return weights and biases as a tuple
        """
        #TODO
        return (self.w, self.b)


class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in  the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should just be a sum of a batch, don't use mean)
        """
        #TODO
        def _softmax(o):
            e = np.exp(o)
            sum = e.sum()
            return e/sum

        def _single_cross_entropy(p, label):
            logp = np.log(p)
            return -np.dot(label.T, logp)
        logits = logits.T
        labels = labels.T
        (num_classes,batch_size) = np.shape(logits)
        self.batch_size = batch_size
        logits = np.apply_along_axis(_softmax, 0, logits)
        self.y_hat = logits
        self.y = labels
        cross_entropy_outcome = [_single_cross_entropy(logits[:,i], labels[:,i]) for i in range(batch_size)]
        if not get_predictions:
            return sum(cross_entropy_outcome)/self.batch_size
        else: 
            return sum(cross_entropy_outcome)/self.batch_size, np.argmax(self.y_hat, axis = 0)


    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't divide by batch_size here in order to pass autograding)
        Transepose might lead to bug!!
        """
        #TODO
        return (self.y_hat - self.y).T

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        #TODO
        max_pred_values = np.max(self.y_hat, axis=0)
        mask_col = (self.y_hat == max_pred_values)
        result = mask_col.astype(int)
        acc = np.sum(result*self.y)
        return acc/self.batch_size



class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SoftMaxCrossEntropy object
        """
        #TODO
        self.Conv = Conv((3,32,32), (1,5,5))
        self.ReLU = ReLU()
        dimH1 = ((32 - 5 +2*2)//1) + 1
        dimW1 = ((32 - 5 +2*2)//1) + 1
        self.MaxPool = MaxPool((2,2), 2)
        self.Flatten = Flatten()
        dimH2 =((dimH1 - 2)//2) + 1
        dimW2 = ((dimW1 - 2)//2) + 1
        self.Linear = LinearLayer(dimH2*dimW2*1, 10)
        self.SoftMax = SoftMaxCrossEntropyLoss()


    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        #TODO
        ret = self.Conv.forward(inputs)
        ret = self.ReLU.forward(ret)
        ret = self.MaxPool.forward(ret)
        ret = self.Flatten.forward(ret)
        ret = self.Linear.forward(ret)
        ret, predicted = self.SoftMax.forward(ret, y_labels, True)
        return ret, predicted, self.SoftMax.getAccu()

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        #TODO
        dloss = self.SoftMax.backward()
        (dw, db, dloss) = self.Linear.backward(dloss)
        dloss = self.Flatten.backward(dloss)
        dloss = self.MaxPool.backward(dloss)
        dloss = self.ReLU.backward(dloss)
        dloss = self.Conv.backward(dloss)

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        #TODO
        self.Conv.update(learning_rate, momentum_coeff)
        self.Linear.update(learning_rate, momentum_coeff)
        
       


class ConvNetTwo:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool ->Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU,LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        
        
        """
        #TODO
        raise NotImplementedError


    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        #TODO
        raise NotImplementedError


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        #TODO
        raise NotImplementedError
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        #TODO
        raise NotImplementedError

class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    (Conv -> Relu -> MaxPool -> Dropout)x3 -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU, Conv, ReLU, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with 16 filters of size 3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 16 filters of size 3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 16 filters of size 3x3
        then apply Relu
        then apply Dropout with probability 0.1
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        #TODO
        pad = 2
        stride = 1
        # 1
        self.Conv1 = Conv((3,32,32), (16,3,3))
        H1 = (32-3+2*pad)+1
        W1 = (32-3+2*pad)+1
        self.ReLU1 = ReLU()
        self.MaxPool1 = MaxPool((2,2), 2)
        H2 = ((H1-2)//2)+1
        W2 = ((W1-2)//2)+1
        self.Dropout1 = Dropout(0.1)
        # 2
        self.Conv2 = Conv((16,H2,W2), (16,3,3))
        H3 = (H2-3+2*pad)+1
        W3 = (W2-3+2*pad)+1
        self.ReLU2 = ReLU()
        self.MaxPool2 = MaxPool((2,2), 2)
        H4 = ((H3-2)//2)+1
        W4 = ((W3-2)//2)+1
        self.Dropout2 = Dropout(0.1)
        # 3
        self.Conv3 = Conv((16,H4,W4), (16,3,3))
        H5 = (H4-3+2*pad)+1
        W5 = (W4-3+2*pad)+1
        self.ReLU3 = ReLU()
        self.MaxPool3 = MaxPool((2,2), 2)
        H6 = ((H5-2)//2)+1
        W6 = ((W5-2)//2)+1
        self.Dropout3 = Dropout(0.1)
        # Linear
        self.Flatten = Flatten()
        # print(H1,H2,H3,H4,H5,H6)
        self.Linear = LinearLayer(H6*W6*16, 10)
        self.SoftMax = SoftMaxCrossEntropyLoss()







    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        #TODO
        ret = self.Conv1.forward(inputs)
        # print(ret.shape)
        ret = self.ReLU1.forward(ret)
        ret = self.MaxPool1.forward(ret)
        # print(ret.shape)
        ret = self.Dropout1.forward(ret)

        ret = self.Conv2.forward(ret)
        # print(ret.shape)
        ret = self.ReLU2.forward(ret)
        ret = self.MaxPool2.forward(ret)
        # print(ret.shape)
        ret = self.Dropout2.forward(ret)

        ret = self.Conv3.forward(ret)
        ret = self.ReLU3.forward(ret)
        # print(ret.shape)
        ret = self.MaxPool3.forward(ret)
        ret = self.Dropout3.forward(ret)
        # print(ret.shape)

        ret = self.Flatten.forward(ret)
        ret = self.Linear.forward(ret)

        ret, predicted = self.SoftMax.forward(ret, y_labels, True)
        return ret, predicted, self.SoftMax.getAccu()


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        #TODO
        dloss = self.SoftMax.backward()
        (dw, db, dloss) = self.Linear.backward(dloss)
        dloss = self.Flatten.backward(dloss)
        # print("here4")

        dloss = self.Dropout3.backward(dloss)
        dloss = self.MaxPool3.backward(dloss)
        dloss = self.ReLU3.backward(dloss)
        _, _, dloss = self.Conv3.backward(dloss)
        # print("here5")

        dloss = self.Dropout2.backward(dloss)
        dloss = self.MaxPool2.backward(dloss)
        dloss = self.ReLU2.backward(dloss)
        _, _, dloss = self.Conv2.backward(dloss)
        # print("here6")

        dloss = self.Dropout1.backward(dloss)
        dloss = self.MaxPool1.backward(dloss)
        dloss = self.ReLU1.backward(dloss)
        _, _, dloss = self.Conv1.backward(dloss)
        # print("here1")
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        #TODO
        self.Conv1.update(learning_rate, momentum_coeff)
        self.Conv2.update(learning_rate, momentum_coeff)
        self.Conv3.update(learning_rate, momentum_coeff)
        self.Linear.update(learning_rate, momentum_coeff)
        


def one_hot_encode(labels):
    """
    One hot encode labels
    """
    one_hot_labels = np.array([[i==label for i in range(len(CLASS_IDS.keys()))] for label in labels], np.int32)
    return one_hot_labels


def prep_imagenet_data(train_images, train_labels, val_images, val_labels):
    # onehot encode labels
    train_labels = one_hot_encode(train_labels)
    val_labels = one_hot_encode(val_labels)

    # standardize to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    val_images = (val_images - 127.5) / 127.5

    # put channels first
    train_images = np.transpose(train_images, (0, 3, 1, 2))
    val_images = np.transpose(val_images, (0, 3, 1, 2))

    return train_images, train_labels, val_images, val_labels

# 1a
def DataVis(train_dataset, train_labels, val_dataset, val_labels, num_sample_per_class):
    return 0
# 1b
def DataStas(dataset, labels, val_dataset, val_labels):
    # 1: number of samples per class
    # 2: data type, data range(min, max), data mean per image channel
    # 3: data standard deviation

    # training data: 
    # Task 1: 
    # Use numpy.bincount to count the occurrences
    counts = np.bincount(labels)
    # The counts array will contain the counts for each value
    for value, count in enumerate(counts):
        if count > 0:
            print(f"Class {value} appears {count} times")
    # Task 2: 
    return 0


# 2a
class Plotting():
    def __init__(self, model, n_epochs, lr, momentum, batch_size, seed, training_data,training_labels, testing_data, testing_labels):
        self.model = model
        self.n_epochs = n_epochs
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.seed = seed
        self.training_data = training_data
        self.testing_data = testing_data
        self.training_labels = training_labels
        self.testing_labels = testing_labels


        self.train_prec_loss =  [0.]*self.n_epochs
        self.test_prec_loss =  [0.]*self.n_epochs
        self.train_prec_accuracy =  [0.]*self.n_epochs
        self.test_prec_accuracy =  [0.]*self.n_epochs
    
    def training(self):
        (train_sample_size, channel, height, weight) =  self.training_data.shape
        for n in range(self.n_epochs):
            print(n)
            for curr_batch in range(0, train_sample_size, self.batch_size):
                if curr_batch+self.batch_size > train_sample_size:
                    break
                batch_train = self.training_data[curr_batch:(curr_batch+self.batch_size),:,:,:]
                batch_train_labels = self.training_labels[curr_batch:(curr_batch+self.batch_size), :]
                ret, predicted, accuracy = self.model.forward(batch_train, batch_train_labels)
                self.model.backward()
                self.model.update(self.lr, self.momentum)
            loss, predicted, accuracy = self.model.forward(self.training_data, self.training_labels)
            self.train_prec_loss[n] = loss
            self.train_prec_accuracy[n] = accuracy

            loss, predicted, accuracy  = self.model.forward(self.testing_data, self.testing_labels)
            self.test_prec_loss[n] = loss
            self.test_prec_accuracy[n] = accuracy
    
    def plot_curves(self):
        plt.figure(figsize=(12, 5))

        # Subplot 1 for accuracy
        plt.subplot(1, 2, 1)
        # setting 1 
        plt.plot(range(self.n_epochs), self.train_prec_accuracy, label='Training Accuracy', color='b', linestyle='-')
        plt.plot(range(self.n_epochs), self.test_prec_accuracy, label='Testing Accuracy', color='g', linestyle='-')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()


        plt.subplot(1, 2, 2)
        # setting 1 
        plt.plot(range(self.n_epochs), self.train_prec_loss, label='Training Loss', color='b', linestyle='-')
        plt.plot(range(self.n_epochs), self.test_prec_loss, label='Testing Loss', color='g', linestyle='-')
        
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()



# Implement the training as you wish. This part will not be autograded
# Feel free to implement other helper libraries (i.e. matplotlib, seaborn) but please do not import other libraries (i.e. torch, tensorflow, etc.) for the training
#Note: make sure to download the data from the resources tab on piazza
if __name__ == '__main__':
    # This part may be helpful to write the training loop
    
    # Training parameters
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--momentum', type=float, default = 0.95)
    parser.add_argument('--num_epochs', type=int, default = 50)
    parser.add_argument('--seed', type=int, default = 47)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--name_prefix', type=str, default=None)
    parser.add_argument('--num_filters', type=int, default=16)
    parser.add_argument('--filter_size', type=int, default=3)
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    MOMENTUM = args.momentum
    EPOCHS = args.num_epochs
    SEED = args.seed
    # print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    ## DATA EXPLORATION 
    with open("10417-tiny-imagenet-train-bal.pkl", "rb") as f:
        train_dict = pickle.load(f)
        train_images = train_dict["images"]
        train_labels = train_dict["labels"]
    
    with open("10417-tiny-imagenet-val-bal.pkl", "rb") as f:
        val_dict = pickle.load(f)
        val_images = val_dict["images"]
        val_labels = val_dict["labels"]
    
    ## Problem 1a: Data Vis
    # TODO: plot data samples for train/val
    

    ## Problem 1b: Data Statistics
    # TODO: plot/show image stats

    # preprocessing imagenet data for training (don't change this)
    train_images, train_labels, val_images, val_labels = prep_imagenet_data(train_images, train_labels, val_images, val_labels)

    ## Problem 2a: Train ConvNet
    np.random.seed(SEED)
    # TODO
    # Conv2a = ConvNet()
    # P2A = Plotting(Conv2a, EPOCHS, LEARNING_RATE, MOMENTUM, BATCH_SIZE, SEED, train_images, train_labels, val_images, val_labels)
    # P2A.training()
    # P2A.plot_curves()
    
    

    ## Problem 2b: Train ConvNetThree
    # np.random.seed(SEED)
    # TODO
    Conv2b = ConvNetThree()
    P2B = Plotting(Conv2b, EPOCHS, LEARNING_RATE, MOMENTUM, BATCH_SIZE, SEED, train_images, train_labels, val_images, val_labels)
    P2B.training()
    P2B.plot_curves()
    

    ## Problem 2c: Train your best model
    # np.random.seed(SEED)
    # TODO
    
    ## Problem 3a: Evaluation
    # TODO: plot confusion matrix and misclassified images on imagenet data

    ## Problem 3b: Evaluate on COCO  "10417-coco.pkl"
    # TODO: Load COCO Data

    # TODO: plot COCO data

    # TODO: get/plot stats COCO

    # TODO: preprocess COCO data, standardize, onehot encode, put channels first
    # hint: see see prep_imagenet_data() for reference (make sure data range is [-1, 1] before eval!)
    
    # TODO: get loss and accuracy COCO

    # TODO: get confusion matrix COCO and misclassified images COCO
