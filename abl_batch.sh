CUDA_VISIBLE_DEVICES=1 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=1 --metric=IB --use_norm=0   &
CUDA_VISIBLE_DEVICES=1 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=2 --metric=IB --use_norm=0   &
CUDA_VISIBLE_DEVICES=1 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=4 --metric=IB --use_norm=0   &
CUDA_VISIBLE_DEVICES=1 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=8 --metric=IB --use_norm=0   &
CUDA_VISIBLE_DEVICES=1 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=16 --metric=IB --use_norm=0   &
CUDA_VISIBLE_DEVICES=2 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=32 --metric=IB --use_norm=0   &
CUDA_VISIBLE_DEVICES=2 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=64 --metric=IB --use_norm=0   &
CUDA_VISIBLE_DEVICES=2 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0 --batchsize=128 --metric=IB --use_norm=0   &
wait

