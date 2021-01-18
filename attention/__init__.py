import sys
import mxnet as mx
from mxnet.gluon import nn

from attention.eca import ECA
from attention.gct import GCT
from attention.se import SE
from attention.se_v1 import SEv1
from attention.se_v2 import SEv2
from attention.se_v3 import SEv3


class Attention(nn.HybridBlock):
    def __init__(self, attention_type, channels=1, **kwargs):
        super(Attention, self).__init__(**kwargs)
        with self.name_scope():
            attention_type = attention_type.lower()
            if attention_type == 'eca':
                self.attention = ECA()
            elif attention_type == 'gct':
                self.attention = GCT(channels)
            elif attention_type == 'se':
                self.attention = SE(channels)
            elif attention_type == 'se-v1':
                self.attention = SEv1(channels)
            elif attention_type == 'se-v2':
                self.attention = SEv2(channels)
            elif attention_type == 'se-v3':
                self.attention = SEv3(channels)
            else:
                sys.exit('Unsupported channel attention type: %s' % attention_type)

    def hybrid_forward(self, F, x):
        return self.attention(x)
