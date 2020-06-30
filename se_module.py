import mxnet as mx
from mxnet import nd

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class SE(nn.HybridBlock):
    def __init__(self, channels=1, r=16, **kwargs):
        super(SE, self).__init__(**kwargs)
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.fc1 = nn.Dense(channels // r, use_bias=False)
            self.relu = nn.Activation('relu')
            self.fc2 = nn.Dense(channels, use_bias=False)
            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # squeeze
        y = self.fc1(y)
        y = self.relu(y)
        # excitate
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = F.reshape(y, (-2, 1, 1))
        return F.broadcast_mul(x, y)
