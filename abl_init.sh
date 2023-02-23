CUDA_VISIBLE_DEVICES=1 python ablation_init.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=128 --metric=IB --use_norm=0 --init_method=kaiming_norm_fanin &
CUDA_VISIBLE_DEVICES=1 python ablation_init.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=128 --metric=IB --use_norm=0 --init_method=kaiming_norm_fanout &
CUDA_VISIBLE_DEVICES=3 python ablation_init.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=128 --metric=IB --use_norm=0 --init_method=kaiming_uniform_fanin &
CUDA_VISIBLE_DEVICES=3 python ablation_init.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=128 --metric=IB --use_norm=0 --init_method=kaiming_uniform_fanout &
CUDA_VISIBLE_DEVICES=3 python ablation_init.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=128 --metric=IB --use_norm=0 --init_method=xavier_uniform &
CUDA_VISIBLE_DEVICES=3 python ablation_init.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=128 --metric=IB --use_norm=0 --init_method=xavier_normal &
wait
