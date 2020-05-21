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
<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">Vanilla<br></th>
    <th colspan="2">ECA</th>
  </tr>
  <tr>
    <td>validation loss<br></td>
    <td>validation acc</td>
    <td>validation loss</td>
    <td>validation acc</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>cifar_resnet20_v1</td>
    <td align="center">0.0344</span></td>
    <td align="center">0.9171</span></td>
    <td align="center">0.0302</span></td>
    <td align="center">0.9189</span></td>
  </tr>
  <tr>
    <td>cifar_resnet20_v2</span></td>
    <td align="center">0.1088</span></td>
    <td align="center">0.9133</span></td>
    <td align="center">0.0328</span></td>
    <td align="center">0.9194</span></td>
  </tr>
  <tr>
    <td>cifar_resnet56_v1</span></td>
    <td align="center">0.0431</span></td>
    <td align="center">0.9154</span></td>
    <td align="center">0.0170</span></td>
    <td align="center">0.9243</span></td>
  </tr>
  <tr>
    <td>cifar_resnet56_v2</span></td>
    <td align="center">0.0629</span></td>
    <td align="center">0.9165</span></td>
    <td align="center">0.0235</span></td>
    <td align="center">0.9218</span></td>
  </tr>
</tbody>
</table>
