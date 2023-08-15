import torch
from numpy import newaxis as np_newaxis

# TODO: Please be sure to read the comments in the main lab and think about your design before
# you begin to implement this part of the lab.

# Layers in this file are arranged in roughly the order they
# would appear in a network.


"""
    You should implement the __init__ methods of all the classes,
    the Network’s set_input and forward methods, and the layer 
    class's clear_grad, set, randomize, and forward methods.  
    Each layer should maintain the instance variables output, 
    which represents the output of that layer.
"""

class Layer:
    def __init__(self, output_shape, outputs=None, train=False):
        """
        TODO: Add arguments and initialize instance attributes here.
        """
        self.output_shape = output_shape
        self.outputs = outputs
        self.train = train
        self.grad = 0

    def accumulate_grad(self, x):
        """
        TODO: Add arguments as needed for this method.
        This method should accumulate its grad attribute with the value provided.
        """
        
        self.grad += x
        
            

    def clear_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        self.grad = 0

    def step(self):
        """
        TODO: Add arguments as needed for this method.
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass

class Input(Layer):
    def __init__(self, output_shape, outputs, train=False):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, output_shape, None, train) # TODO: Pass along any arguments to the parent's initializer here.
        if outputs is not None:
            assert output_shape == outputs.shape, print('outputs.shape and output_shape should match.')
            self.outputs = outputs



    def set(self,outputs):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer.
        :param output: The output to set, as a torch tensor. Raise an error if this output's size
                       would change.
        """
        print(outputs.shape)
        print(self.output_shape)
        assert outputs.shape[1] == self.output_shape[1], print('Output shape passed in should be the same as output shape specified')
        self.outputs = outputs



    def randomize(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer to random values sampled from the standard normal
        distribution (torch has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.randn(self.output_shape)

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        assert self.output is not None, print("You need to set the output of the linear layer")
        return self.output

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self):
        """
        TODO: Add arguments as needed for this method.
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """
        pass

class Linear(Layer):
    def __init__(self, output_shape, input_shape, inputs, weights=None, biases=None, train=False):
        """
        TODO: Accept any arguments specific to this child class.
        """
        
        
        self.inputs = inputs
        Layer.__init__(self, output_shape, None, train) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_shape = input_shape
        
        if weights is None:
            self.weights = Layer((input_shape[1], output_shape[1]), torch.randn(input_shape[1], output_shape[1]), True)
        else:

            assert weights.output_shape == (self.input_shape[1], self.output_shape[1]), print('weights are incorrect shape passed in to linear layer')
            self.weights = weights
        if biases is None:
            self.biases = Layer(output_shape, torch.randn(output_shape), True)
        else:

            assert biases.output_shape == output_shape, print('Biases shape should match output_shape')
            self.biases = biases
        self.outputs = None
        
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        assert self.inputs.outputs.shape[1] == self.input_shape[1], "Input is not the right shape for a linear layer"
        self.outputs = self.inputs.outputs@self.weights.outputs + self.biases.outputs
        
        

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        
        self.dj_dw = self.inputs.outputs.T@self.grad
        self.dj_dx = self.grad@self.weights.outputs.T
        
        self.dj_db = self.grad.sum(axis=0, keepdim=True)
        

        
        self.weights.accumulate_grad(self.dj_dw)
        self.inputs.accumulate_grad(self.dj_dx)
        self.biases.accumulate_grad(self.dj_db)


class ReLU(Layer):
    def __init__(self, output_shape, inputs):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, output_shape, None, False) # TODO: Pass along any arguments to the parent's initializer here.
        
        self.inputs = inputs
        
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        assert self.inputs.outputs.shape[1] == self.output_shape[1], print('Inputs must match the shape of the ReLU layer')
        self.outputs = torch.max(torch.zeros_like(self.inputs.outputs), self.inputs.outputs)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        
        self.dj_dw = self.grad * (self.inputs.outputs > 0)
        self.inputs.accumulate_grad(self.dj_dw)


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    def __init__(self, inputs):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, (1), None, False) # TODO: Pass along any arguments to the parent's initializer here.
        self.inputs = inputs
        
    
    def set_target(self, target):
        self.target = target
        
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """

        self.outputs = torch.mean((self.inputs.outputs - self.target.outputs)**2)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.dj_do = 2 * (self.inputs.outputs - self.target.outputs) * self.grad
        self.inputs.accumulate_grad(self.dj_do)

class Regularization(Layer):
    def __init__(self, matrix_shape, inputs, lam=1):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, (1), None, False) # TODO: Pass along any arguments to the parent's initializer here.
        self.matrix_shape = matrix_shape
        self.inputs = inputs
        self.lam = lam
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        assert self.inputs.outputs.shape == self.matrix_shape, print('The input shape is incorrect')
        
        self.outputs = self.lam * torch.sum(torch.square(self.inputs.outputs))

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.dj_dw = self.lam * 2 * self.inputs.outputs
        self.inputs.accumulate_grad(self.dj_dw)


class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    TODO: Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self, inputs):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, (1), None, False) # TODO: Pass along any arguments to the parent's initializer here.
        self.inputs = inputs
    
    def set_target(self, target):
        self.target = target
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        assert self.inputs.outputs.shape == self.target.outputs.shape, print('output needs to match input shape')
        epsilon = 1e-8
        
        m = torch.max(self.inputs.outputs, dim=1, keepdim=True)
        
        temp = self.inputs.outputs - m.values
        
        exp_outputs = torch.exp(temp)
        
        self.classifications = exp_outputs / exp_outputs.sum(dim=1, keepdim=True)
        L = (self.target.outputs * torch.log(self.classifications + epsilon)).sum() / self.classifications.shape[0]
        self.outputs = -L
        if self.outputs <= 0:
            print('inputs: ',temp)
            print('classifications: ', self.classifications)
            

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.dj_dv = self.classifications - self.target.outputs
        self.inputs.accumulate_grad(self.dj_dv)


class Sum(Layer):
    def __init__(self, a, b):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, (1), None, False) # TODO: Pass along any arguments to the parent's initializer here.
        self.a = a
        self.b = b

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        
        self.outputs = self.a.outputs + self.b.outputs

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        # Since a and b are only used once accumulate grad for a will always be 1
        self.dj_da = 1
        self.dj_db = 1

        # THis is only needed when an input is used in multiple layers
        self.a.accumulate_grad(self.dj_da)
        self.b.accumulate_grad(self.dj_db)
        


