import numpy as np
import sys

sys.path.append("mytorch")
from gru_cell import *
from linear import *


class CharacterPredictor(object):

    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        """The network consists of a GRU Cell and a linear layer."""
        self.hidden_dim = hidden_dim
        self.rnn = GRUCell(input_dim, hidden_dim)
        self.projection = Linear(hidden_dim, num_classes)
        self.projection.W = np.random.rand(num_classes, hidden_dim)

    def init_rnn_weights(
        self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
    ):
        # DO NOT MODIFY
        self.rnn.init_weights(
            Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
        )

    def __call__(self, x, h):
        return self.forward(x, h)

    def show_hidden_dim(self):
        return self.hidden_dim

    def forward(self, x, h):
        """CharacterPredictor forward.

        A pass through one time step of the input

        Input
        -----
        x: (feature_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        logits: (num_classes)
            hidden state at current time-step.

        hnext: (hidden_dim)
            hidden state at current time-step.

        """
        hnext = self.rnn.forward(x, h)
        # self.projection expects input in the form of batch_size * input_dimension
        # Therefore, reshape the input of self.projection as (1,-1)
        logits = self.projection.forward(hnext.reshape(1, -1))
        logits = logits.reshape(-1,)
        return logits, hnext

def inference(net, inputs):
    """CharacterPredictor inference.

    An instance of the class defined above runs through a sequence of inputs to
    generate the logits for all the timesteps.

    Input
    -----
    net:
        An instance of CharacterPredictor.

    inputs: (seq_len, feature_dim)
            a sequence of inputs of dimensions.

    Returns
    -------
    logits: (seq_len, num_classes)
            one per time step of input..

    """
    hidden_dim = net.show_hidden_dim()
    sequences_out = []
    hnext = np.zeros(hidden_dim)
    for input in inputs:
        logits, hnext = net.forward(input, hnext)
        sequences_out.append(logits)

    return np.array(sequences_out)
