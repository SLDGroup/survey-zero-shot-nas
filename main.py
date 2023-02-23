from typing import Dict
from nas_201_api import NASBench201API as API201
import models
from models import NB101Network
import datasets
from nats_bench import create as create_nats
import os
import argparse
import torch 
import torch.nn as nn
import random 
import numpy as np
from nasbench import api as api101
from ptflops import get_model_complexity_info
from measures import get_grad_score, get_ntk_n, get_batch_jacobian, get_logdet, get_zenscore


parser = argparse.ArgumentParser(description='ZS-NAS')
parser.add_argument('--searchspace', metavar='ss', type=str, choices=['101','201','nats','nats_tss', 'mbnv2', 'resnet'],
                    help='define the target search space of benchmark')
parser.add_argument('--dataset', metavar='ds', type=str, choices=['cifar10','cifar100','ImageNet16-120','imagenet-1k', 'cifar10-valid'],
                    help='select the dataset')
parser.add_argument('--data_path', type=str, default='~/dataset/',
                    help='the path where you store the dataset')
parser.add_argument('--cutout', type=int, default=0,
                    help='use cutout or not on input data')
parser.add_argument('--batchsize', type=int, default=1024,
                    help='batch size for each input batch')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of threads for data pipelining')
parser.add_argument('--metric', type=str, choices=['basic','ntk','lr', 'logdet', 'grad', 'zen','IB'],
                    help='define the zero-shot proxy for evaluation')
parser.add_argument('--startnetid', type=int, default=0,
                    help='the index of the first network to be evaluated in the search space. currently only works for nb101')
parser.add_argument('--manualSeed', type=int, default=0,
                    help='random seed')
args = parser.parse_args()


def getmisc(args):
    manualSeed=args.manualSeed
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.dataset == "cifar10":
        root = args.data_path
        imgsize=32
    elif args.dataset == "cifar100":
        root = args.data_path
        imgsize=32
    elif args.dataset.startswith("imagenet-1k"):
        root = args.data_path+'ILSVRC/Data/CLS-LOC'
        imgsize=224
    elif args.dataset.startswith("ImageNet16"):
        root = args.data_path+'img16/ImageNet16/'
        imgsize=16
    
    train_data, test_data, xshape, class_num = datasets.get_datasets(args.dataset, root, args.cutout)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsize, shuffle=True, num_workers=args.num_worker)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batchsize, shuffle=False, num_workers=args.num_worker)

    ce_loss = nn.CrossEntropyLoss().cuda()
    # filename = 'misc/'+'{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(args.metric, args.searchspace, args.dataset,args.batchsize, \
    #                                 args.cutout, args.gamma1, args.gamma2, args.maxbatch)
    return imgsize, ce_loss, trainloader, testloader


def search201(api, netid, dataset):
    if dataset=='cifar10':
        dsprestr='ori'
    else: 
        dsprestr='x'
    results = api.query_by_index(netid, dataset, hp= '200') 
    train_loss, train_acc, test_loss, test_acc =0, 0, 0, 0
    for seed, result in results.items():
        train_loss += result.get_train()['loss']
        train_acc += result.get_train()['accuracy']
        test_loss += result.get_eval(dsprestr+'-test')['loss']
        test_acc += result.get_eval(dsprestr+'-test')['accuracy']
    config = api.get_net_config(netid, dataset)
    network = models.get_cell_based_tiny_net(config) 
    num_trails = len(results)
    train_loss, train_acc, test_loss, test_acc = \
            train_loss/num_trails, train_acc/num_trails, test_loss/num_trails, test_acc/num_trails
    return network, [train_acc, train_acc, test_loss, test_acc]


def search_nats(api, netid, dataset, hpval):
    # Simulate the training of the 1224-th candidate:
    # validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(netid, dataset=dataset, hp=hpval)
    testacc = api.get_more_info(netid, dataset, hp=hpval)['test-accuracy']
    config = api.get_net_config(netid, dataset)
    network = models.get_cell_based_tiny_net(config)
    return network, testacc


def get101acc(data_dict:dict):
    acc4=(data_dict[4][0]['final_test_accuracy']+data_dict[4][1]['final_test_accuracy']+data_dict[4][2]['final_test_accuracy'])/3.0
    acc12=(data_dict[12][0]['final_test_accuracy']+data_dict[12][1]['final_test_accuracy']+data_dict[12][2]['final_test_accuracy'])/3.0
    acc36=(data_dict[36][0]['final_test_accuracy']+data_dict[36][1]['final_test_accuracy']+data_dict[36][2]['final_test_accuracy'])/3.0
    acc108=(data_dict[108][0]['final_test_accuracy']+data_dict[108][1]['final_test_accuracy']+data_dict[108][2]['final_test_accuracy'])/3.0
    return [acc4,acc12,acc36,acc108]
    

def get_basic(netid, network, metrics, imgsize, space = 'cv'):
    if space == 'cv':
        macs, params = get_model_complexity_info(network, (3, imgsize, imgsize), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
        return netid, macs, params, metrics


def enumerate_networks(args):
    imgsize, ce_loss, trainloader, testloader = getmisc(args)

    if '101' in args.searchspace.lower():
        assert args.dataset == "cifar10"
        NASBENCH_TFRECORD = '~/dataset/nasbench/nasbench_full.tfrecord'
        nasbench = api101.NASBench(NASBENCH_TFRECORD)

        def getallacc(data_dict:dict):
            acc4=sum(data_dict[4][i]['final_test_accuracy'] for i in range(3))/3.0
            acc12=sum(data_dict[12][i]['final_test_accuracy'] for i in range(3))/3.0
            acc36=sum(data_dict[36][i]['final_test_accuracy'] for i in range(3))/3.0
            acc108=sum(data_dict[108][i]['final_test_accuracy'] for i in range(3))/3.0
            return [acc4,acc12,acc36,acc108]

        if args.metric in ['logdet', 'grad']:
            for i, batch in enumerate(trainloader):
                data,label = batch[0],batch[1]
                data,label=data.cuda(),label.cuda()
                break

        allnethash = list(nasbench.hash_iterator())
        for netid in range(args.startnetid, len(allnethash)):
            unique_hash = allnethash[netid]
            fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
            acc_metrics= getallacc(computed_metrics)
            
            ops = fixed_metrics['module_operations']
            adjacency = fixed_metrics['module_adjacency']

            network = NB101Network((adjacency, ops))
            network.cuda()
            if args.metric =='basic':
                get_basic(netid, network, acc_metrics, imgsize)
            elif args.metric =='ntk':
                get_ntk_n(netid, trainloader, network, train_mode=True, num_batch=1)
            elif args.metric =='logdet':
                get_logdet(netid, network, args, data, label)
            elif args.metric =='zen':
                get_zenscore(netid, network, imgsize, args.batchsize)
            elif args.metric =='grad':
                get_grad_score(netid, network,  data, label, ce_loss, split_data=1, device='cuda')
        
                
    elif '201' in args.searchspace.lower():
        api = API201('~/dataset/nasbench/NAS-Bench-201-v1_1-096897.pth', verbose=False)

        if args.metric in ['logdet', 'grad']:
            for i, batch in enumerate(trainloader):
                data,label = batch[0],batch[1]
                data,label=data.cuda(),label.cuda()
                break

        for netid in range(1000000):
            network, metric = search201(api, netid, args.dataset)
            network.cuda()

            if args.metric =='basic':
                get_basic(netid, network, metric, imgsize)
            elif args.metric =='ntk':
                get_ntk_n(netid, trainloader, network, train_mode=True, num_batch=1)
            elif args.metric =='logdet':
                get_logdet(netid, network, args, data, label)
            elif args.metric =='zen':
                get_zenscore(netid, network, imgsize, args.batchsize)
            elif args.metric =='grad':
                get_grad_score(netid, network,  data, label, ce_loss, split_data=1, device='cuda')
                
                
    elif 'nats' in args.searchspace.lower():
        if 'tss' in args.searchspace.lower():
            # Create the API instance for the topology search space in NATS
            api = create_nats('~/dataset/nasbench/NATS/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=True)
            hpval='200'
        else:
            # Create the API instance for the size search space in NATS
            api = create_nats('~/dataset/nasbench/NATS/NATS-sss-v1_0-50262-simple', 'sss', fast_mode=True, verbose=True)
            hpval='90'

        if args.metric in ['logdet', 'grad']:
            for i, batch in enumerate(trainloader):
                data,label = batch[0],batch[1]
                data,label=data.cuda(),label.cuda()
                break

        for netid in range(1000000):
            network, metric = search_nats(api, netid, args.dataset, hpval)
            network.cuda()
            if args.metric =='basic':
                get_basic(netid, network, [metric], imgsize)
            elif args.metric =='ntk':
                get_ntk_n(netid, trainloader, network, train_mode=True, num_batch=1)
            elif args.metric =='logdet':
                get_logdet(netid, network, args, data, label)
            elif args.metric =='zen':
                get_zenscore(netid, network, imgsize, args.batchsize)
            elif args.metric =='grad':
                get_grad_score(netid, network,  data, label, ce_loss, split_data=1, device='cuda')

if __name__ == 'main':
    enumerate_networks(args)




