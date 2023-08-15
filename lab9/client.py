
# +
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
import os
from cool_adam import Adam
import matplotlib.pyplot as plt


# warnings.filterwarnings('ignore')  # If you see warnings that you know you can ignore, it can be useful to enable this.

EPOCHS = 1
# For simple regression problem
TRAINING_POINTS = 1000

# For fashion-MNIST and similar problems
DATA_ROOT = '/data/cs3450/data/'
FASHION_MNIST_TRAINING = '/data/cs3450/data/fashion_mnist_flattened_training.npz'
FASHION_MNIST_TESTING = '/data/cs3450/data/fashion_mnist_flattened_testing.npz'
CIFAR10_TRAINING = '/data/cs3450/data/cifar10_flattened_training.npz'
CIFAR10_TESTING = '/data/cs3450/data/cifar10_flattened_testing.npz'
CIFAR100_TRAINING = '/data/cs3450/data/cifar100_flattened_training.npz'
CIFAR100_TESTING = '/data/cs3450/data/cifar100_flattened_testing.npz'

# With this block, we don't need to set device=DEVICE for every tensor.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.cuda.set_device(0)
     torch.set_default_tensor_type(torch.cuda.FloatTensor)
     print("Running on the GPU, its cuda time!")
else:
     print("Running on the CPU")
        
        
# makes plots nice
def make_pretty(ax, title='', x_label='', y_label='', img=False):
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    if img:
        ax.axis('off')
    return ax



# -

def load_dataset_flattened(train=True,dataset='Fashion-MNIST',download=False):
    """
    :param train: True for training, False for testing
    :param dataset: 'Fashion-MNIST', 'CIFAR-10', or 'CIFAR-100'
    :param download: True to download. Keep to false afterwords to avoid unneeded downloads.
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    if dataset == 'Fashion-MNIST':
        if train:
            path = FASHION_MNIST_TRAINING
        else:
            path = FASHION_MNIST_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-10':
        if train:
            path = CIFAR10_TRAINING
        else:
            path = CIFAR10_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-100':
        if train:
            path = CIFAR100_TRAINING
        else:
            path = CIFAR100_TESTING
        num_labels = 100
    else:
        raise ValueError('Unknown dataset: '+str(dataset))

    if os.path.isfile(path):
        print('Loading cached flattened data for',dataset,'training' if train else 'testing')
        data = np.load(path)
        x = torch.tensor(data['x'],dtype=torch.float32)
        y = torch.tensor(data['y'],dtype=torch.float32)
        pass
    else:
        class ToTorch(object):
            """Like ToTensor, only redefined by us for 'historical reasons'"""

            def __call__(self, pic):
                return torchvision.transforms.functional.to_tensor(pic)

        if dataset == 'Fashion-MNIST':
            data = torchvision.datasets.FashionMNIST(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-10':
            data = torchvision.datasets.CIFAR10(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-100':
            data = torchvision.datasets.CIFAR100(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        else:
            raise ValueError('This code should be unreachable because of a previous check.')
        x = torch.zeros((len(data[0][0].flatten()), len(data)),dtype=torch.float32)
        for index, image in enumerate(data):
            x[:, index] = data[index][0].flatten()
        labels = torch.tensor([sample[1] for sample in data])
        y = torch.zeros((num_labels, len(labels)), dtype=torch.float32)
        y[labels, torch.arange(len(labels))] = 1
        np.savez(path, x=x.numpy(), y=y.numpy())
    return x, y







# +
def acc(pred, target):
    target_labels = torch.argmax(target, dim=1)
    pred_labels = torch.argmax(pred, dim=1)
    num_correct = torch.sum(torch.eq(target_labels, pred_labels))
    acc = float(num_correct) / float(pred.shape[0])
    
    return acc



def SGD(inputs, outputs, epochs, batch_size, learning_rate, network, test_inputs, test_outputs):
    accs = []
    losses = []
    test_accs = []
    test_losses = []
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for sample in range(0, inputs.shape[0], batch_size):

            x.set(inputs[sample:sample+batch_size,:])
            target.set(outputs[sample:sample+batch_size,:])
            network.forward()
            network.backward()
            network.step(learning_rate)
            
            
        print(f'Loss: {network.layers[-1].outputs}')
        x.set(inputs)
        target.set(outputs)
        network.forward()
        outs_forward_pass = network.layers[-2].classifications
        acc_score = acc(outs_forward_pass, target.outputs)
        losses.append(network.layers[-1].outputs.cpu().numpy())
        accs.append(acc_score)
        
        x.set(test_inputs)
        target.set(test_outputs)
        network.forward()
        outs_forward_test_pass = network.layers[-2].classifications
        ta = acc(outs_forward_test_pass, target.outputs)
        tl = network.layers[-1].outputs.cpu().numpy()
        test_accs.append(ta)
        test_losses.append(tl)
        
        print(acc_score)
    return accs, losses, test_accs, test_losses
    

def Adam_loop(inputs, outputs, epochs, batch_size, network, optimizer):
    accs = []
    losses = []
    
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for sample in range(0, inputs.shape[0], batch_size):

            x.set(inputs[sample:sample+batch_size,:])
            target.set(outputs[sample:sample+batch_size,:])
            network.forward()
            network.backward()
            optimizer.step()
            
            
        print(f'Loss: {network.layers[-1].outputs}')
        x.set(inputs)
        target.set(outputs)
        network.forward()
        outs_forward_pass = network.layers[-2].classifications
        acc_score = acc(outs_forward_pass, target.outputs)
        accs.append(acc_score)
        losses.append(network.layers[-1].outputs)
        print(acc_score)
    return accs, losses




# -










# ### FashionMNIST network

inputs, outputs = load_dataset_flattened()
test_inputs, test_outputs = load_dataset_flattened(train=False)

w = Layer((784, 100), torch.rand(784,100) * 0.1, True)
b = Layer((1, 100), torch.rand(1,100) * 0.1, True)
m = Layer((100, 10), torch.rand(100,10) * 0.1, True)
c = Layer((1, 10), torch.rand(1,10) * 0.1, True)
x = Input((1, 784), torch.rand(1,784))
target = Input((1, 10), torch.Tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]))
# One Linear Layer
Linear1 = Linear((1, 100), (1, 784), x, w, b)

# One ReLU layer
re = ReLU((1, 100), Linear1)

Linear2 = Linear((1, 10), (1, 100), re, m, c)


Reg1 = Regularization((784, 100), w, lam=0)
Reg2 = Regularization((100, 10), m, lam=0)

Sum1 = Sum(Reg1, Reg2)

so = Softmax(Linear2)

Sum2 = Sum(so, Sum1)

network = Network()

network.add_parameter(w)
network.add_parameter(b)
network.add_parameter(m)
network.add_parameter(c)
network.set_input(x)
so.set_target(target)

network.add_layer(Linear1)
network.add_layer(re)
network.add_layer(Linear2)
network.add_layer(Reg1)
network.add_layer(Reg2)
network.add_layer(Sum1)
network.add_layer(so)
network.add_layer(Sum2)





inputs = inputs.T
outputs = outputs.T
test_inputs = test_inputs.T
test_outputs = test_outputs.T
print(inputs.shape)
print(outputs.shape)
print(test_inputs.shape)
print(test_outputs.shape)

# # Fashion MNIST train

# +
accs_fmnist, losses_fmnist, test_accs_fmnist, test_losses_fmnist = \
SGD(inputs, outputs, 20, 5, 0.01, network, test_inputs, test_outputs)





# +
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(np.arange(1, 21), test_accs_fmnist, label='test accuracy')
ax.plot(np.arange(1, 21), accs_fmnist, label='train accuracy')


make_pretty(ax, 'Training / Test: Accuracy  - FMNIST', 'Epoch', 'Score')

# +
fig, ax = plt.subplots(figsize=(15, 10))

ax.plot(np.arange(1, 21), losses_fmnist, label='train loss')

ax.plot(np.arange(1, 21), test_losses_fmnist, label='test loss')
make_pretty(ax, 'Training / Test: Loss - FMNIST', 'Epoch', 'Score')
# -

# ### Cifar10

inputs, outputs = load_dataset_flattened(train=True, dataset='CIFAR-10')
test_inputs, test_outputs = load_dataset_flattened(train=False, dataset='CIFAR-10')
inputs = inputs.T
outputs = outputs.T
test_inputs = test_inputs.T
test_outputs = test_outputs.T
inputs.shape


# +
w = Layer((3072, 250), torch.rand(3072, 250) * 0.1, True)
b = Layer((1, 250), torch.rand(1,250) * 0.1, True)
m = Layer((250, 10), torch.rand(250,10) * 0.1, True)
c = Layer((1, 10), torch.rand(1,10) * 0.1, True)
x = Input((1, 3072), torch.rand(1,3072))
target = Input((1, 10), torch.Tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]))
# One Linear Layer
Linear1 = Linear((1, 250), (1, 3072), x, w, b)


# One ReLU layer
re = ReLU((1, 250), Linear1)


Linear2 = Linear((1, 10), (1, 250), re, m, c)

Reg1 = Regularization((3072, 250), w, lam=0)
Reg2 = Regularization((250, 10), m, lam=0)

Sum1 = Sum(Reg1, Reg2)
so = Softmax(Linear2)
Sum2 = Sum(so, Sum1)

network = Network()

network.add_parameter(w)
network.add_parameter(b)
network.add_parameter(m)
network.add_parameter(c)
network.set_input(x)
so.set_target(target)

network.add_layer(Linear1)
network.add_layer(re)
network.add_layer(Linear2)
network.add_layer(Reg1)
network.add_layer(Reg2)
network.add_layer(Sum1)
network.add_layer(so)
network.add_layer(Sum2)


print(inputs.shape)
print(outputs.shape)

# +
accs_cifar, losses_cifar, test_accs_cifar, test_losses_cifar = \
SGD(inputs, outputs, 100, 5, 0.0001, network, test_inputs, test_outputs)




# +
fig, ax = plt.subplots(figsize=(15, 10))

ax.plot(np.arange(1, 101), accs_cifar, label='train accuracy')
ax.plot(np.arange(1, 101), test_accs_cifar, label='test accuracy')
make_pretty(ax, 'Training / Test: Accuracy - CIFAR-10', 'Epoch', 'Score')

# -


fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(np.arange(1, 101), losses_cifar, label='train loss')
ax.plot(np.arange(1, 101), test_losses_cifar, label='test loss')
make_pretty(ax, 'Training / Test: Loss - CIFAR-10', 'Epoch', 'Score')

# +
# I implemented an Adam optimizer instead SGD, but it is complaining about my layer objects
# optimizer = Adam([x.outputs for x in network.parameters], 0.01)
# Adam_loop(inputs, outputs, 10, 5, network, optimizer)
# -
test_losses_cifar[-1]


test_accs_cifar[-1]

test_accs_fmnist[-1]


