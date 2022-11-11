import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets

###
### Sequential MNIST
###
class seqMNIST():
    def __init__(self, par):
        # Get datasets
        mnist_trainset = datasets.MNIST(
            root='./data', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(
            root='./data', train=False, download=True, transform=None)
        '''
        # Using keras and tensorflow
        from keras.datasets import mnist 
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        self.X_te = torch.from_numpy(test_X)
        y_te = torch.zeros(10000,dtype=int)
        y_te[:] = torch.from_numpy(test_y)
        self.X_tr = torch.from_numpy(train_X)
        y_tr = torch.zeros(60000,dtype=int)
        y_tr[:] = torch.from_numpy(train_y)
        '''
        self.X_te = mnist_testset.data  # Test set images
        y_te = mnist_testset.targets  # Test set labels
        N_te = y_te.size()[0]  # Number of test samples
        self.N_o = 10 # No. output units/classes
        
        # Initialisation of the one-hot encoded labels for the test set
        self.Y_te = torch.zeros([N_te, self.N_o])
        # From labels to one-hot encoded labels for the test set
        self.Y_te[np.arange(0, N_te), y_te] = 1

        self.X_tr = mnist_trainset.data  # Train set images
        y_tr = mnist_trainset.targets  # Train labels
        N_tr = y_tr.size()[0]

        # Initialisation of one-hot encoded labels for training
        self.Y_tr = torch.zeros([N_tr, self.N_o])
        # From labels to one-hot encoded labels for the training set
        self.Y_tr[np.arange(0, N_tr), y_tr] = 1

        i_val = np.random.permutation(np.arange(0, N_tr))[0:par.N_val]

        self.X_val = self.X_tr[i_val, :, :]
        self.Y_val = self.Y_tr[i_val, :]

        i_tr = np.delete(np.arange(0, N_tr), i_val)
        N_tr = N_tr-par.N_val

        self.X_tr = self.X_tr[i_tr, :, :]
        self.Y_tr = self.Y_tr[i_tr, :]

        self.T = self.X_tr.size()[2]
        self.N_i = self.X_tr.size()[1]

        # Normalisation and conversion to float
        X_M = 255

        self.X_tr = self.X_tr.float()/X_M
        self.X_val = self.X_val.float()/X_M
        self.X_te = self.X_te.float()/X_M

        self.Y_tr = self.Y_tr.float()
        self.Y_val = self.Y_val.float()
        self.Y_te = self.Y_te.float()
