# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn.functional as F
import torch.nn as nn

from . import measure
# from ..p_utils import get_layer_metric_array


def get_layer_metric_array(net, metric, mode): 
    metric_array = []

    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    
    return metric_array



@measure('plain', bn=True, mode='param')
def compute_plain_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, space='cv'):

    net.zero_grad()
    if space == 'asr':
        N = inputs[0].shape[0]
        inputdata, inputlen = inputs[0], inputs[1]
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            fmz, outputs = net.forward(inputdata[st:en])
            loss = loss_fn(outputs, inputlen[st:en]//4, targets[0][st:en], targets[1][st:en])
            loss.backward()
    else: 
        N = inputs.shape[0]
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            fmz, outputs = net.forward(inputs[st:en])
            loss = loss_fn(outputs, targets[st:en])
            loss.backward()

    # select the gradients that we want to use for search/prune
    def plain(layer):
        if layer.weight.grad is not None:
            return layer.weight.grad * layer.weight
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, plain, mode)
    return grads_abs
