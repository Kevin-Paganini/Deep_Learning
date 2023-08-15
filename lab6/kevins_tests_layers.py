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
from layers import MSELoss

class Kevin_Layers_Test(TestCase):
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
        
        self.ReLU2 = ReLU((1, 2), self.Linear2)
        
        self.Reg1 = Regularization((2, 3), self.w)
        self.Reg2 = Regularization((3, 2), self.m)
        
        self.Sum = Sum(self.Reg1, self.Reg2)

        self.Softmax = Softmax(self.Linear2)
        self.Softmax.set_target(self.target)
    
        self.MSELoss = MSELoss(self.Linear2)
        self.MSELoss.set_target(self.target)
        
        

        
    
    
    def test_sum(self):
        print('Sum test:')
        self.Reg1.forward()
        self.Reg2.forward()
        self.Sum.forward()
            
    def test_reg_1(self):
        print('Reg 1 test:')
        self.Reg1.forward()
        print(self.Reg1.outputs) 
        np.testing.assert_allclose(self.Reg1.outputs.numpy(), np.array([67]))
        
        
    def test_reg_2(self):
        print('Reg 2 test:')
        self.Reg2.forward()
        print(self.Reg2.outputs) 
        np.testing.assert_allclose(self.Reg2.outputs.numpy(), np.array([122]))

    def test_linear(self):
        print('Startin lienar layer test')
        self.Linear1.forward()
        
        print(self.Linear1.outputs)
        np.testing.assert_allclose(self.Linear1.outputs.numpy(), np.array([[16, 21, 19]]))
      
    def test_Linear_ReLU(self):
        print('Starting Linear ReLU test')
        self.Linear1.forward()
        self.ReLU.forward()
        print(self.ReLU.outputs)
        np.testing.assert_allclose(self.ReLU.outputs.numpy(), np.array([[16, 21, 19]]))
        
        
    def test_two_ReLUs(self):
        print('Starting two linear layer test')
        self.Linear1.forward()
        self.ReLU.forward()
        self.Linear2.forward()
        self.ReLU2.forward()
        print(self.ReLU2.outputs)
        np.testing.assert_allclose(self.ReLU2.outputs.numpy(), np.array([[5, 8]]))


    def test_softmax(self):
        print('Starting Softmax test')
        self.Linear1.forward()
        self.ReLU.forward()
        self.Linear2.forward()
        print(self.Linear2.outputs)
        self.Softmax.forward()
        print(self.Softmax.outputs)
        np.testing.assert_allclose(self.Softmax.outputs.numpy(), np.array([0.04858]), atol=1e-4)

    def test_MSE(self):
        print('Starting Softmax test')
        self.Linear1.forward()
        self.ReLU.forward()
        self.Linear2.forward()
        print(self.Linear2.outputs)
        self.MSELoss.forward()
        print(self.MSELoss.outputs)
        np.testing.assert_allclose(self.MSELoss.outputs.numpy(), np.array([37]), atol=1e-4)




if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)