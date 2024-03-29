import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):

        self.conv1 = Conv1d(24, 8, 8, 4)
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16, 4, 1, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def init_weights(self, weights):

        w1,w2,w3 = weights
        self.conv1.conv1d_stride1.W = np.transpose(w1.T.reshape(8, 8, 24), (0, 2, 1))
        self.conv2.conv1d_stride1.W = np.transpose(w2.T.reshape(16, 1, 8), (0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(w3.T.reshape(4, 1, 16), (0, 2, 1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):

        self.conv1 = Conv1d(24, 2, 2, 2)
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8, 4, 2, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):
        return self.forward(A)

    def init_weights(self, weights):

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = np.transpose(w1[:48, :2].T.reshape(2, 2, 24), (0, 2, 1))
        self.conv2.conv1d_stride1.W = np.transpose(w2[:4, :8].T.reshape(8, 2, 2), (0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(w3.T.reshape(4, 2, 8), (0, 2, 1))


    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
