import mxnet as mx
from mxnet import nd

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class GCT(nn.HybridBlock):
    def __init__(self, channels=1, beta_wd_mult=0.0, epsilon=1e-5, **kwargs):
        super(GCT, self).__init__(**kwargs)
        self.epsilon = epsilon
        with self.name_scope():
            self.alpha = self.params.get('alpha', shape=(1,channels,1,1), init=mx.init.Constant(1.0))
            self.beta = self.params.get('beta', shape=(1,channels,1,1), init=mx.init.Constant(0.0), wd_mult=beta_wd_mult)
            self.gamma = self.params.get('gamma', shape=(1,channels,1,1), init=mx.init.Constant(0.0))

    def hybrid_forward(self, F, x, alpha, beta, gamma):
        embedding = F.broadcast_mul(F.sqrt(F.sum(F.square(x), axis=(2,3), keepdims=True) + self.epsilon),
                                    alpha)
        norm = F.sqrt(F.mean(F.square(embedding), axis=1, keepdims=True) + self.epsilon)
        embedding = F.broadcast_div(embedding, norm)
        gate = 1.0 + F.tanh(F.broadcast_add(F.broadcast_mul(gamma, embedding), beta))
        return F.broadcast_mul(x, gate)
