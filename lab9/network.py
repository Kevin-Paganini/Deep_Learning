class Network:
    def __init__(self):
        """
        TODO: Initialize a `layers` attribute to hold all the layers in the gradient tape.
        """
        self.layers = []
        self.loss = None
        self.parameters = []
        self.input = None


    def add_layer(self, layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        self.layers.append(layer)

    def add_parameter(self, parameter):
        """

        Adds a new parameter to the network
        All parameters need to be added before something like a linear layer
        """
        self.parameters.append(parameter)

    def set_loss(self, loss):
        self.loss = loss

    def set_input(self,input):
        """
        :param input: The sublayer that represents the signal input (e.g., the image to be classified)
        """
        # TODO: Delete or implement this method. (Implementing this method is optional, but do not
        # leave it as a stub.)
        self.input = input

    def set_output(self,output):
        """
        :param output: SubLayer that produces the useful output (e.g., clasification decisions) as its output.
        """
        # TODO: Delete or implement this method. (Implementing this method is optional, but do not
        # leave it as a stub.)
        #
        # This becomes messier when your output is the variable o from the middle of the Softmax
        # layer -- I used try/catch on accessing the layer.classifications variable.
        # when trying to access read the output layer's variable -- and that ended up being in a
        # different method than this one.

    def forward(self):
        """
        Compute the output of the network in the forward direction, working through the gradient
        tape forward

        :param input: A torch tensor that will serve as the input for this forward pass
        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        # TODO: Implement this method
        # TODO: Either remove the input option and output options, or if you keep them, assign the
        #  input to the input layer's output before performing the forward evaluation of the network.
        #
        # Users will be expected to add layers to the network in the order they are evaluated, so
        # this method can simply call the forward method for each layer in order.
        for layer in self.layers:
            layer.forward()
        
        return self.layers[-1].outputs

    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation backward through the 
        gradient tape.

        """
        for layer in self.layers[::-1]:
            layer.backward()

    def step(self, lr):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward, updating all learnable parameters 

        """
        
        
        for i, param in enumerate(self.parameters):
            
            # print(param.outputs.shape)
            param.outputs -= lr * param.grad
        
        for x in self.layers:
            x.grad = 0
            
        for x in self.parameters:
            x.grad = 0
             

###################################
# Output
# w@m tensor([[-1.4680e-02,  9.9837e-01],
#            [-9.9638e-01,  4.1951e-04]])
# I use row vectors instead of column vectors
# ##################################        
