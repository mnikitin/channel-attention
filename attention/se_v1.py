import mxnet as mx
from mxnet import nd

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class SEv1(nn.HybridBlock):
    def __init__(self, channels=1, **kwargs):
        super(SEv1, self).__init__(**kwargs)
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):
        y = self.avg_pool(x)
        y = self.sigmoid(y)
        return F.broadcast_mul(x, y)

