import os
import torch
import torchvision
import numpy as np

# For simple regression problem
TRAINING_POINTS = 1000

def create_linear_training_data():
    """
    This method simply rotates points in a 2D space.
    Be sure to use MSE in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a numpy array where columns are training samples and
             y is a numpy array where columns are one-hot labels for the training sample.
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
    :return: (x,y) the dataset. x is a numpy array where columns are training samples and
             y is a numpy array where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].copy()
    x2 = x[1:2, :]
    x2 *= 2 * ((x2 > 0) - 0.5)
    y = np.concatenate((-x2, x1), axis=0)
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


if __name__ == '__main__':
    # The code in this section should NOT be in a helper method.
    # But you may choose to occassionally move helper methods before this as
    # the code within them becomes stable.
    #
    # For this week's lab, however, you can likely keep ALL your code
    # right here, with all your variables in the global scope 
    # for simplified debugging.

    # TODO: You may wish to make each TODO below its own pynb cell.
    # TODO: Build your network.

    # TODO: Select your datasource.
    x_train, y_train = create_linear_training_data()

    # TODO: Train your network.

    # TODO: Sanity-check the output of your network.
    # You can optionally compute the error on this test data:
    x_test, y_test = create_linear_training_data()

    # But you must computed W*M as discussed in the lab assignment.
    x_test, y_test = create_linear_training_data()

    pass # You may wish to keep this line as a point to place a debugging breakpoint.
