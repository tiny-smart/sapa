# On Point Affiliation in Feature Upsampling

Code for [On Point Affiliation in Feature Upsampling](https://arxiv.org/abs/2307.08198), which is an extension for the NeurIPS 2022 paper [SAPA: Similarity-Aware Point Affiliation for Feature Upsampling](https://arxiv.org/abs/2209.12866).

## Installation

The code is tested on Python 3.8.8 and PyTorch 1.9.0.

For the base module (SAPA-B),
```shell
cd sapa
python setup.py develop
```
And for SAPA with dynamic selection (SAPA-D),
```shell
cd deformsapa
python setup.py develop
```
## Results

#### Segformer-B1
| Segformer-B1 | FLOPs | Params | mIoU      | bIoU  |
| :--:         | :--:  | :--:   | :--:      | :--:  |
| Bilinear     | 15.9  | 13.74  | 42.11     | 28.16 |  
| CARAFE       | +1.5  | +0.44  | 42.82     | 29.84 |
| IndexNet     | +30.7 | +12.6  | 41.50     | 28.27 |
| A2U          | +0.4  | +0.1   | 41.45     | 27.31 |
| FADE         | +2.7  | +0.3   | 43.06     | *31.68* |
| SAPA-I       | +0.8  | +0     | 43.05     | 30.25 |
| SAPA-B       | +1.0  | +0.1   | *43.20*   | 30.96 | 
| SAPA-D       | +5.3  | +0.8   | **44.68** | 31.74 |
