# CUDA_VISIBLE_DEVICES=0 python ablationmean.py --searchspace=201 --dataset=cifar100 --cutout=0  --metric=IB   &
# CUDA_VISIBLE_DEVICES=1 python ablationmean.py --searchspace=201 --dataset=ImageNet16-120 --cutout=0  --metric=IB  &
# CUDA_VISIBLE_DEVICES=2 python ablationmean.py --searchspace=nats --dataset=cifar100 --cutout=0  --metric=IB   &
# CUDA_VISIBLE_DEVICES=3 python ablationmean.py --searchspace=nats --dataset=ImageNet16-120 --cutout=0  --metric=IB    &
# wait


# CUDA_VISIBLE_DEVICES=0 python ablationstd.py --searchspace=201 --dataset=cifar100 --cutout=0  --metric=IB   &
# CUDA_VISIBLE_DEVICES=1 python ablationstd.py --searchspace=201 --dataset=ImageNet16-120 --cutout=0  --metric=IB  &
# CUDA_VISIBLE_DEVICES=2 python ablationstd.py --searchspace=nats --dataset=cifar100 --cutout=0  --metric=IB   &
# CUDA_VISIBLE_DEVICES=3 python ablationstd.py --searchspace=nats --dataset=ImageNet16-120 --cutout=0  --metric=IB    &
# wait

# CUDA_VISIBLE_DEVICES=2 python ablationmean.py --searchspace=201 --dataset=cifar10 --cutout=0  --metric=IB   &
# CUDA_VISIBLE_DEVICES=3 python ablationstd.py --searchspace=201 --dataset=cifar10 --cutout=0  --metric=IB  &
# wait

CUDA_VISIBLE_DEVICES=1 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0  --metric=IB  --batchsize=256 &
CUDA_VISIBLE_DEVICES=0 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0  --metric=IB  --batchsize=512   &
CUDA_VISIBLE_DEVICES=2 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0  --metric=IB  --batchsize=1024 &
CUDA_VISIBLE_DEVICES=3 python ablation.py --searchspace=201 --dataset=cifar100 --cutout=0  --metric=IB  --batchsize=4096   &
wait

