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

import copy

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



@measure('grad_norm', bn=True)
def get_grad_norm_arr(net, inputs, targets, loss_fn, split_data=1, skip_grad=False, space='cv'):
    net.zero_grad()
    if space == 'asr':
        N = inputs[0].shape[0]
        inputdata, inputlen = inputs[0], inputs[1]
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data
            
            fmz, outputs = net.forward(inputdata[st:en])
            loss = loss_fn(outputs, inputlen[st:en]//4, targets[0][st:en], targets[1][st:en])    
            # print(loss)
            # exit()
            loss.backward()
            grad_norm_arr = get_layer_metric_array(net, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')
    else: 
        N = inputs.shape[0]
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            fmz, outputs = net.forward(inputs[st:en])
            loss = loss_fn(outputs, targets[st:en])
            # print(loss)
            # exit()
            loss.backward()

            grad_norm_arr = get_layer_metric_array(net, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')
    return grad_norm_arr
