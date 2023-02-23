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
import torch.nn as nn
import numpy as np
from .utils import *
from . import grad_norm
from . import snip
from . import grasp
from . import fisher
from . import jacob_cov
from . import plain
from . import synflow

available_measures = []
_measure_impls = {}


def measure(name, bn=True, copy_net=False, force_clean=True, **impl_args):
    def make_impl(func):
        def measure_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net_orig
            ret = func(net, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc
                import torch
                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _measure_impls
        if name in _measure_impls:
            raise KeyError(f'Duplicated measure! {name}')
        available_measures.append(name)
        _measure_impls[name] = measure_impl
        return func
    return make_impl


def calc_measure(name, net, device, *args, **kwargs):
    return _measure_impls[name](net, device, *args, **kwargs)


def enum_gradient_measure(net, device, *args, **kwargs):
    def sum_arr(arr):
        sum = 0.
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()
    score_list = []
    score_list.append(sum_arr(calc_measure('grad_norm', net, device, *args, **kwargs)))
    score_list.append(sum_arr(calc_measure('snip', net, device, *args, **kwargs)))
    if kwargs['space'] =='cv':
        score_list.append(sum_arr(calc_measure('grasp', net, device, *args, **kwargs)))
    score_list.append(sum_arr(calc_measure('fisher', net, device, *args, **kwargs)))
    score_list.append(calc_measure('jacob_cov', net, device, *args, **kwargs))
    score_list.append(sum_arr(calc_measure('plain', net, device, *args, **kwargs)))
    score_list.append(sum_arr(calc_measure('synflow', net, device, *args, **kwargs)))
    return score_list


def get_ntk_n(dataloader, network, train_mode=False, num_batch=-1):
    device = torch.cuda.current_device()
    ntks = []
    network.to(device)
    networks=[network]
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(dataloader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)[1]
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds[0]


def get_batch_jacobian( net, x, target, ):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()


def get_logdet(network, args, data, label):
    network.cuda()

    network.K = np.zeros((args.batchsize, args.batchsize))
    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
        except:
            pass
    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True
    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            #hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)
    
    jacobs, labels, y, out = get_batch_jacobian(network, data, label)
    s, logdet_val = np.linalg.slogdet(network.K)
    return logdet_val


def network_weight_gaussian_init(net):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def get_zenscore(model, resolution, batch_size, repeat=1, mixup_gamma=1e-2, fp16=False, space = 'cv'):
    info = {}
    nas_score_list = []

    device = torch.device('cuda:0')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32
    if space == 'asr':
        inputsize = resolution
    else:
        inputsize = [batch_size, 3, resolution, resolution]
    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            
            input = torch.randn(size=inputsize, device=device, dtype=dtype)
            input2 = torch.randn(size=inputsize, device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            output, logits = model(input, outpreap=True)
            mixup_output, logits = model(mixup_input, outpreap=True)
            print(output.size())
            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))


    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    scorelist=[info[keyname] for keyname in info.keys() ]
    return scorelist[0],scorelist[1],scorelist[2]


def get_grad_score(network,  data, label, loss_fn, split_data=1, device='cuda', space='cv'):
    score_list = enum_gradient_measure(network, device, data, label, loss_fn=loss_fn, split_data=split_data, space=space)
    


