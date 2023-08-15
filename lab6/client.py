
import numpy as np
import torch
import unittest
from layers import Layer
from layers import Input
from layers import Linear
from layers import ReLU
from layers import Regularization
from layers import Sum
from layers import Softmax
from network import Network
from layers import MSELoss
import torch


def SGD(inputs, outputs, epochs, batch_size, learning_rate):
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for sample in range(0, inputs.shape[0], batch_size):

            x.set(inputs[sample:sample+batch_size,:])
            target.set(outputs[sample:sample+batch_size,:])
            network.forward()
            network.backward()
            # print('before:',network.parameters[0].outputs)
            
            network.step(learning_rate)
            # print('after',network.parameters[0].outputs)
            print(f'Loss: {network.layers[-1].outputs}')
    print(f'w@m {network.parameters[0].outputs@network.parameters[2].outputs}')




def create_linear_training_data():
    """
    This method simply rotates points in a 2D space.
    Be sure to use L2 regression in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_folded_training_data():
    """
    This method introduces a single non-linear fold into the sort of data created by create_linear_training_data. Be sure to REMOVE the final softmax layer before testing on this data!
    Be sure to use MSE in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    x2 *= 2 * ((x2 > 0).float() - 0.5)
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_square():
    """
    This is a square example in which the challenge is to determine
    if the points are inside or outside of a point in 2d space.
    insideness is true if the points are inside the square.
    :return: (points, insideness) the dataset. points is a 2xN array of points and insideness is true if the point is inside the square.
    """
    win_x = [2,2,3,3]
    win_y = [1,2,2,1]
    win = torch.tensor([win_x,win_y],dtype=torch.float32)
    win_rot = torch.cat((win[:,1:],win[:,0:1]),axis=1)
    t = win_rot - win # edges tangent along side of poly
    rotation = torch.tensor([[0, 1],[-1,0]],dtype=torch.float32)
    normal = rotation @ t # normal vectors to each side of poly
        # torch.matmul(rotation,t) # Same thing

    points = torch.rand((2,2000),dtype = torch.float32)
    points = 4*points

    vectors = points[:,np.newaxis,:] - win[:,:,np.newaxis] # reshape to fill origin
    insideness = (normal[:,:,np.newaxis] * vectors).sum(axis=0)
    insideness = insideness.T
    insideness = insideness > 0
    insideness = insideness.all(axis=1)
    return points, insideness

w = Layer((2, 3), torch.rand(2,3) * 0.1, True)
b = Layer((1, 3), torch.rand(1,3) * 0.1, True)
m = Layer((3, 2), torch.rand(3,2) * 0.1, True)
c = Layer((1, 2), torch.rand(1,2) * 0.1, True)
x = Input((1, 2), torch.rand(1,2))
target = Input((1, 2), torch.Tensor([[0, 1]]))
# One Linear Layer
Linear1 = Linear((1, 3), (1, 2), x, w, b)

# One ReLU layer
ReLU = ReLU((1, 3), Linear1)

Linear2 = Linear((1, 2), (1, 3), ReLU, m, c)


Reg1 = Regularization((2, 3), w, lam=0)
Reg2 = Regularization((3, 2), m, lam=0)

Sum1 = Sum(Reg1, Reg2)

Softmax = MSELoss(Linear2)

Sum2 = Sum(Softmax, Sum1)

network = Network()

network.add_parameter(w)
network.add_parameter(b)
network.add_parameter(m)
network.add_parameter(c)
network.set_input(x)
Softmax.set_target(target)

network.add_layer(Linear1)
network.add_layer(ReLU)
network.add_layer(Linear2)
network.add_layer(Reg1)
network.add_layer(Reg2)
network.add_layer(Sum1)
network.add_layer(Softmax)
network.add_layer(Sum2)



TRAINING_POINTS = 1000
inputs, outputs = create_linear_training_data()

inputs = inputs.T
outputs = outputs.T
print(inputs.shape)
print(outputs.shape)

SGD(inputs, outputs, 10, 10, 0.01)


        
        


