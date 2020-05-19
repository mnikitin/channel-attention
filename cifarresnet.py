# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument
"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['get_model', 'get_cifar_resnet',
           'cifar_resnet20_v1', 'cifar_resnet56_v1', 'cifar_resnet110_v1',
           'cifar_resnet20_v2', 'cifar_resnet56_v2', 'cifar_resnet110_v2']

import os
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet import cpu, gpu

from eca_module import ECA

# Helpers
def _conv3x3(channels, stride, in_channel):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channel)


# Blocks
class CIFARBasicBlockV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 norm_layer=BatchNorm, norm_kwargs=None,
                 use_eca=False, **kwargs):
        super(CIFARBasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.eca = ECA() if use_eca else None
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.body(x)
        if self.eca:
            x = self.eca(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = F.Activation(x + residual, act_type='relu')
        return x

class CIFARBasicBlockV2(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 norm_layer=BatchNorm, norm_kwargs=None, 
                 use_eca=False, **kwargs):
        super(CIFARBasicBlockV2, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.eca = ECA() if use_eca else None
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.body(x)
        if self.eca:
            x = self.eca(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = x + residual
        return x

# Nets
class CIFARResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, classes=10,
                 norm_layer=BatchNorm, norm_kwargs=None,
                 use_eca=False, **kwargs):
        super(CIFARResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.use_eca = use_eca
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                            use_eca=self.use_eca, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                use_eca=self.use_eca, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class CIFARResNetV2(HybridBlock):
    def __init__(self, block, layers, channels, classes=10,
                 norm_layer=BatchNorm, norm_kwargs=None,
                 use_eca=False, **kwargs):
        super(CIFARResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.use_eca = use_eca
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(scale=False, center=False,
                                         **({} if norm_kwargs is None else norm_kwargs)))

            self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                in_channels = channels[i+1]
            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())

            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                            use_eca=self.use_eca, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                use_eca=self.use_eca, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification
resnet_net_versions = [CIFARResNetV1, CIFARResNetV2]
resnet_block_versions = [CIFARBasicBlockV1, CIFARBasicBlockV2]

def _get_resnet_spec(num_layers):
    assert (num_layers - 2) % 6 == 0

    n = (num_layers - 2) // 6
    channels = [16, 16, 32, 64]
    layers = [n] * (len(channels) - 1)
    return layers, channels


# Constructor
def get_cifar_resnet(version, num_layers, ctx=cpu(),
                     root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 6*n+2, e.g. 20, 56, 110, 164.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    
    layers, channels = _get_resnet_spec(num_layers)

    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1]
    net = resnet_class(block_class, layers, channels, **kwargs)
    return net

def cifar_resnet20_v1(**kwargs):
    return get_cifar_resnet(1, 20, **kwargs)

def cifar_resnet56_v1(**kwargs):
    return get_cifar_resnet(1, 56, **kwargs)

def cifar_resnet110_v1(**kwargs):
    return get_cifar_resnet(1, 110, **kwargs)

def cifar_resnet20_v2(**kwargs):
    return get_cifar_resnet(2, 20, **kwargs)

def cifar_resnet56_v2(**kwargs):
    return get_cifar_resnet(2, 56, **kwargs)

def cifar_resnet110_v2(**kwargs):
    return get_cifar_resnet(2, 110, **kwargs)


def get_model(name, **kwargs):
    _models = {
        'cifar_resnet20_v1': cifar_resnet20_v1,
        'cifar_resnet56_v1': cifar_resnet56_v1,
        'cifar_resnet110_v1': cifar_resnet110_v1,
        'cifar_resnet20_v2': cifar_resnet20_v2,
        'cifar_resnet56_v2': cifar_resnet56_v2,
        'cifar_resnet110_v2': cifar_resnet110_v2
    }
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net
