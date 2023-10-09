import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha     = alpha
        self.eps       = 1e-8

        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))

        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))

        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval:
            self.Z         = Z
            self.N         = Z.shape[0]
            self.NZ        = (Z - np.repeat(np.expand_dims(self.running_M, axis=0), self.N, axis=0))/np.repeat(np.expand_dims(np.sqrt(self.running_V + self.eps), axis=0), self.N, axis=0)
            self.BZ        = self.BW * self.NZ + self.Bb
            return self.BZ

        self.Z         = Z
        self.N         = Z.shape[0]

        self.M         = Z.mean(0)
        self.V         = Z.var(0)
        self.NZ        = (Z - np.repeat(np.expand_dims(self.M, axis=0), self.N, axis=0))/np.repeat(np.expand_dims(np.sqrt(self.V + self.eps), axis=0), self.N, axis=0)
        self.BZ        = self.BW * self.NZ + self.Bb

        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW  = np.sum(dLdBZ * self.NZ, axis=0)
        self.dLdBb  = np.sum(dLdBZ, axis=0)

        dLdNZ       = dLdBZ * self.BW
        dLdV        = (-1/2)*np.sum(dLdNZ * (self.Z - np.repeat(np.expand_dims(self.Z.mean(0), axis=0), self.N, axis=0)) * np.repeat(np.expand_dims(((self.Z.var(0) + self.eps)**(-3/2)), axis=0), self.N, axis=0), axis=0)
        dLdM        = - np.sum(dLdNZ * np.repeat(np.expand_dims(((self.Z.var(0) + self.eps)**(-1/2)), axis=0), self.N, axis=0), axis=0) - 2/self.N * dLdV * np.sum((self.Z - np.repeat(np.expand_dims(self.Z.mean(0), axis=0), self.N, axis=0)), axis=0)

        dLdZ        = dLdNZ * np.repeat(np.expand_dims(((self.V + self.eps)**(-1/2)), axis=0), self.N, axis=0) + np.repeat(np.expand_dims(dLdV, axis=0), self.N, axis=0) * (2/self.N * (self.Z - np.repeat(np.expand_dims(self.M, axis=0), self.N, axis=0))) + np.repeat(np.expand_dims(dLdM/self.N, axis=0), self.N, axis=0)

        return  dLdZ
