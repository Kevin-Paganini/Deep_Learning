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
        
        
    def test_network(self):
        self.network.forward()
        print(self.network.layers[-1].outputs)
        np.testing.assert_allclose(self.network.layers[-1].outputs.numpy(), np.array([189.0486]), atol=1e-3)
        self.network.backward()
        # Weights w
        print('Weights w:')
        print(self.network.parameters[0].grad)
        np.testing.assert_allclose(self.network.parameters[0].grad.numpy(), np.array([[3.71269, 6.131842, 7.5668], [5.80846, 10.09228, 3.7112]]), atol=1e-1)
        # bias b
        print('Bias b:')
        print(self.network.parameters[1].grad)
        np.testing.assert_allclose(self.network.parameters[1].grad.numpy(), np.array([[-0.09577, 0.04614, -0.1444]]), atol=1e-1)

        # weights m
        print('Weights m:')
        print(self.network.parameters[2].grad)
        np.testing.assert_allclose(self.network.parameters[2].grad.numpy(), np.array([[4.752, 7.24112], [6.987, 3.00397], [10.893, 15.09883]]), atol=1e-1)
        # bias c
        print('Bias c:')
        print(self.network.parameters[3].grad)
        np.testing.assert_allclose(self.network.parameters[3].grad.numpy(), np.array([[0.047, -0.04743]]), atol=1e-1)
     
     
    def test_step(self):
        self.network.forward()
        print(self.network.layers[-1].outputs)
        np.testing.assert_allclose(self.network.layers[-1].outputs.numpy(), np.array([189.0486]), atol=1e-3)
        self.network.backward()
        self.network.step(0.1)
        # Weights w
        print('Weights w:')
        print(self.network.parameters[0].outputs)
        np.testing.assert_allclose(self.network.parameters[0].outputs.numpy(), np.array([[1.628731, 2.3868, 3.2433], [2.419154, 3.9907, 1.62888]]), atol=1e-1)
        # bias b
        print('Bias b:')
        print(self.network.parameters[1].outputs)
        np.testing.assert_allclose(self.network.parameters[1].outputs.numpy(), np.array([[4.009577, 1.995386, 3.01444]]), atol=1e-1)

        # weights m
        print('Weights m:')
        print(self.network.parameters[2].outputs)
        np.testing.assert_allclose(self.network.parameters[2].outputs.numpy(), np.array([[1.5248, 3.27588], [2.3013, 1.699603], [3.9107, 6.490117]]), atol=1e-1)
        # bias c
        print('Bias c:')
        print(self.network.parameters[3].outputs)
        np.testing.assert_allclose(self.network.parameters[3].outputs.numpy(), np.array([[-185.0047, -249.9952]]), atol=1e-1)
     
     
        
        

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)