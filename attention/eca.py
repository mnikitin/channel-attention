import mxnet as mx
from mxnet import nd

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class ECA(nn.HybridBlock):
    def __init__(self, k_size=3, **kwargs):
        super(ECA, self).__init__(**kwargs)
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.conv = nn.Conv1D(1, kernel_size=k_size, padding=(k_size - 1) // 2,
                                  in_channels=1, use_bias=False)
            self.sigmoid = nn.Activation('sigmoid')


    def hybrid_forward(self, F, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        
        # Two different branches of ECA module        
        y = F.transpose(F.squeeze(y, -1), (0, 2, 1))
        y = self.conv(y)
        y = F.expand_dims(F.transpose(y, (0, 2, 1)), axis=-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return F.broadcast_mul(x, y)
