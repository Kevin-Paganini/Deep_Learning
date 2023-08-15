from unittest import TestCase
import layers
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

class Kevin_Network_Tests(TestCase):
    """
    Please note: I (Dr. Yoder) may have assumed different parameters for my network than you use.
    TODO: Update these tests to work with YOUR definitions of arguments and variables.
    """
    def setUp(self):
        # learnable parameters
        self.w = Layer((2, 3), torch.Tensor([[2, 3, 4],[3, 5, 2]]), True)
        self.b = Layer((1, 3), torch.Tensor([[4, 2, 3]]), True)
        self.m = Layer((3, 2), torch.Tensor([[2, 4], [3, 2], [5, 8]]), True)
        self.c = Layer((1, 2), torch.Tensor([[-185, -250]]), True)
        self.x = Input((1, 2), torch.Tensor([[3, 2]]))
        self.target = Input((1, 2), torch.Tensor([[0, 1]]))
        # One Linear Layer
        self.Linear1 = Linear((1, 3), (1, 2), self.x, self.w, self.b)
        
        # One ReLU layer
        self.ReLU = ReLU((1, 3), self.Linear1)
        
        self.Linear2 = Linear((1, 2), (1, 3), self.ReLU, self.m, self.c)
        
        
        self.Reg1 = Regularization((2, 3), self.w)
        self.Reg2 = Regularization((3, 2), self.m)
        
        self.Sum = Sum(self.Reg1, self.Reg2)

        self.Softmax = Softmax(self.Linear2)
    
        self.Sum2 = Sum(self.Softmax, self.Sum)

        # added so forward pass is easier
        self.network = Network()
        
        self.network.add_parameter(self.w)
        self.network.add_parameter(self.b)
        self.network.add_parameter(self.m)
        self.network.add_parameter(self.c)
        self.network.set_input(self.x)
        self.Softmax.set_target(self.target)
        
        self.network.add_layer(self.Linear1)
        self.network.add_layer(self.ReLU)
        self.network.add_layer(self.Linear2)
        self.network.add_layer(self.Reg1)
        self.network.add_layer(self.Reg2)
        self.network.add_layer(self.Sum)
        self.network.add_layer(self.Softmax)
        self.network.add_layer(self.Sum2)

        self.network.forward()

        np.testing.assert_allclose(self.network.layers[-1].outputs.numpy(), np.array([189.0486]), atol=1e-3)
        
    def test_sum(self):
        print('\nTest first sum:')
        self.Sum2.backward()
        print(self.Softmax.grad)
        np.testing.assert_allclose(np.array([self.Softmax.grad]), np.array([1]), atol=1e-3)
        print()
        
        
    def test_softmax(self):
        print('\nTest softmax:')
        self.Sum2.backward()
        self.Softmax.backward()
        print(self.Linear2.grad)
        np.testing.assert_allclose(self.Linear2.grad.numpy(), np.array([[0.0474, -0.04743]]), atol=1e-3)
        print()
        
    def test_weights_reg(self):
        print('\nTest regularization of weights:')
        self.Sum2.backward()
        self.Softmax.backward()
        self.Sum.backward()
        self.Reg2.backward()
        self.Reg1.backward()
        
        print(self.m.grad)
        np.testing.assert_allclose(self.m.grad.numpy(), np.array([[4, 8], [6, 4], [10, 16]]), atol=1e-3)
        print(self.w.grad)
        np.testing.assert_allclose(self.w.grad.numpy(), np.array([[4, 6, 8], [6, 10, 4]]), atol=1e-3)
        print()

    def test_accumulate_grad(self):
        print('\nTest accumulation of weights:')
        self.Sum2.backward()
        self.Softmax.backward()
        self.Sum.backward()
        self.Reg2.backward()
        self.Reg1.backward()
        self.Linear2.backward()
        self.ReLU.backward()
        self.Linear1.backward()
        print(self.w.grad)
        print(self.m.grad)
        np.testing.assert_allclose(self.m.grad.numpy(), np.array([[4.752, 7.24112], [6.987, 3.00397], [10.893, 15.09883]]), atol=1e-1)
        
        np.testing.assert_allclose(self.w.grad.numpy(), np.array([[3.71269, 6.131842, 7.5668], [5.80846, 10.09228, 3.7112]]), atol=1e-1)
        print()
        
        
        
    
    
    def test_Linear_test(self):
        print('\nLinear test:')
        self.Sum2.backward()
        self.Softmax.backward()
        self.Sum.backward()
        self.Reg2.backward()
        self.Reg1.backward()
        self.Linear2.backward()
        self.ReLU.backward()
        self.Linear1.backward()
        
        np.testing.assert_allclose(self.ReLU.grad.numpy(), np.array([[-0.09572, 0.04614, -0.1444]]), atol=1e-1)
        print(self.ReLU.grad)
        np.testing.assert_allclose(self.x.grad.numpy(), np.array([[-0.63072, -0.34541]]), atol=1e-1)
        print(self.x.grad)
        print()
    
    
    def test_relu(self):
        print('\nTest ReLU:')
        self.Sum2.backward()
        self.Softmax.backward()
        self.Sum.backward()
        self.Linear2.backward()
        self.ReLU.backward()
        
        
        np.testing.assert_allclose(self.Linear1.grad.numpy(), np.array([[-0.09577, 0.04614, -0.1444]]), atol=1e-1)
        print(self.Linear1.grad)
        print()
    
        
    
    
    
    def test_biases(self):
        print('Test biases:')
        self.Sum2.backward()
        self.Softmax.backward()
        self.Sum.backward()
        self.Linear2.backward()
        self.ReLU.backward()
        self.Linear1.backward()
        
        
        np.testing.assert_allclose(self.b.grad.numpy(), np.array([[-0.09577, 0.04614, -0.1444]]), atol=1e-1)
        print(self.c.grad)
        np.testing.assert_allclose(self.c.grad.numpy(), np.array([[0.047, -0.04743]]), atol=1e-1)
        print(self.c.grad)
    
    
    
     

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)