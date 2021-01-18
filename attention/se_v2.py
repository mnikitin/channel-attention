import mxnet as mx
from mxnet import nd

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class SEv2(nn.HybridBlock):
    def __init__(self, channels=1, **kwargs):
        super(SEv2, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(1, channels, 1, 1),
                                          init=mx.init.Constant(1.0))
            self.avg_pool = nn.GlobalAvgPool2D()
            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, weight):
        y = self.avg_pool(x)
        y = F.broadcast_mul(y, weight)
        y = self.sigmoid(y)
        return F.broadcast_mul(x, y)

