# ECANet
Gluon implementation of ECA-Net: https://arxiv.org/abs/1910.03151<br/>
Based on original PyTorch implementation: https://github.com/BangguWu/ECANet

## CIFAR-10 experiments

### Usage
Example of training *resnet20_v1* with *ECA*:<br/>
```
python3 train_cifar10.py --mode hybrid --num-gpus 1 -j 8 --batch-size 128 --num-epochs 186 --lr 0.003 --lr-decay 0.1 --lr-decay-epoch 81,122 --wd 0.0001 --optimizer adam --random-crop --model cifar_resnet20_v1 --eca
```

### Results
TBA
