import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        self.z1 = np.matmul(self.Wrx, self.x)
        self.z2 = np.matmul(self.Wrh, self.hidden)
        self.z3 = self.z1 + self.z2
        self.z4 = self.z3 + self.brx
        self.z5 = self.z4 + self.brh
        self.r = self.r_act(self.z5)

        self.z6 = np.matmul(self.Wzx, self.x)
        self.z7 = np.matmul(self.Wzh, self.hidden)
        self.z8 = self.z6 + self.z7
        self.z9 = self.z8 + self.bzx
        self.z10 = self.z9 + self.bzh
        self.z = self.r_act(self.z10)

        self.z11 = np.matmul(self.Wnx, self.x)
        self.z12 = np.matmul(self.Wnh, self.hidden)
        self.z13 = self.z12 + self.bnh
        self.z14 = np.multiply(self.r, self.z13)
        self.z15 = self.z11 + self.z14
        self.z16 = self.z15 + self.bnx
        self.n = self.h_act(self.z16)

        self.z17 = 1 - self.z
        self.z18 = np.multiply(self.z17, self.n)
        self.z19 = np.multiply(self.z, self.hidden)
        self.z20 = self.z18 + self.z19
        h_t = self.z20

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        delta = delta.reshape(1, -1)
        hidden = (self.hidden).reshape(-1,1)
        x = self.x.reshape(-1,1)
        dLdz20 = delta
        dLdz19 = dLdz20
        dLdz18 = dLdz20
        dLdz = dLdz19 * hidden.T
        dh = dLdz19 * (self.z).reshape(-1, 1).T
        dLdz17 = dLdz18 * (self.n).reshape(-1, 1).T
        dLdn = dLdz18 * (self.z17).reshape(-1, 1).T
        dLdz += (-dLdz17)

        dLdz16 = dLdn * (1 - (self.h_act(self.z16.reshape(-1, 1)))**2).T
        dLdz15 = dLdz16
        self.dbnx = dLdz16
        dLdz14 = dLdz15
        dLdz11 = dLdz15
        dLdr = dLdz14 * (self.z13).reshape(-1, 1).T
        dLdz13 = dLdz14 * (self.r).reshape(-1, 1).T
        dLdz12 = dLdz13
        self.dbnh = dLdz13
        self.dWnh = np.matmul(hidden, dLdz12).T
        dh += np.matmul(dLdz12, self.Wnh)
        self.dWnx = np.matmul(x, dLdz11).T
        dx = np.matmul(dLdz11, self.Wnx)

        dLdz10 = dLdz * self.r_act(self.z10.reshape(-1, 1)).T * (1 - self.r_act(self.z10.reshape(-1, 1))).T
        dLdz9 = dLdz10
        self.dbzh = dLdz10
        dLdz8 = dLdz9
        self.dbzx = dLdz9
        dLdz7 = dLdz8
        dLdz6 = dLdz8
        self.dWzh = np.matmul(hidden, dLdz7).T
        dh += np.matmul(dLdz7, self.Wzh)
        self.dWzx = np.matmul(x, dLdz6).T
        dx += np.matmul(dLdz6, self.Wzx)

        dLdz5 = dLdr * self.r_act(self.z5.reshape(-1, 1)).T * (1 - self.r_act(self.z5.reshape(-1, 1))).T
        dLdz4 = dLdz5
        self.dbrh = dLdz5
        dLdz3 = dLdz4
        self.dbrx = dLdz4
        dLdz1 = dLdz3
        dLdz2 = dLdz3
        self.dWrh = np.matmul(hidden, dLdz2).T
        dh += np.matmul(dLdz2, self.Wrh)
        self.dWrx = np.matmul(x, dLdz1).T
        dx += np.matmul(dLdz1, self.Wrx)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
