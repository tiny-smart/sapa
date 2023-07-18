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

#### Semantic segmentation on ADE20K
| Segformer-B1 | FLOPs | Params | mIoU      | bIoU      |
| :--:         | :--:  | :--:   | :--:      | :--:      |
| Bilinear     | 15.9  | 13.7   | 42.11     | 28.16     |
| Deconv       | +34.4 | +3.5   | 40.71     | 25.94     |
| PixelShuffle | +34.4 | +3.5   | 41.50     | 26.58     |
| CARAFE       | +1.5  | +0.4   | 42.82     | 29.84     |
| IndexNet     | +30.7 | +12.6  | 41.50     | 28.27     |
| A2U          | +0.4  | +0.1   | 41.45     | 27.31     |
| FADE         | +2.7  | +0.3   | 43.06     | *31.68*   |
| SAPA-I       | +0.8  | +0     | 43.05     | 30.25     |
| SAPA-B       | +1.0  | +0.1   | *43.20*   | 30.96     |
| SAPA-D       | +5.3  | +0.8   | **44.68** | **31.74** |

| UPerNet      | Backbone | Params | mIoU      | bIoU      |
| :--:         | :--:     | :--:   | :--:      | :--:      |
| Bilinear     | R50      | 66.5   | 41.09     | 28.04     |
| Deconv       | R50      | +7.1   | 41.43     | 27.72     |
| PixelShuffle | R50      | +28.3  | 41.35     | 27.49     |
| CARAFE       | R50      | +0.3   | 41.49     | 28.29     |
| IndexNet     | R50      | +25.2  | 41.42     | 27.88     |
| A2U          | R50      | +0.1   | 41.37     | 27.71     |
| FADE         | R50      | +0.1   | *41.83*   | 27.92     |
| SAPA-I       | R50      | +0     | 41.51     | *28.60*   |
| SAPA-B       | R50      | +0.1   | 41.47     | 28.27     |
| SAPA-D       | R50      | +0.1   | **42.60** | **28.97** |
| Bilinear     | R101     | 85.5   | 43.33     | 30.21     |
| SAPA-D       | R50      | +0.1   | **44.31** | **31.47** |

#### Object detection on COCO
| Faster R-CNN | Backbone | Params | $AP$     | $AP_{50}$ | $AP_{75}$ | $AP_S$   | $AP_M$   | $AP_{L}$ |
| :---:        |  :---:   | :---:  | :---:    | :---:     | :---:     | :---:    | :---:    | :---:    | 
| Nearest      | R50      | 46.8   | 37.4     | 58.1      | 40.4      | 21.2     | 41.0     | 48.1     |
| Deconv       | R50      | +2.4   | 37.3     | 57.8      | 40.3      | 21.3     | 41.1     | 48.0     |
| PixelShuffle | R50      | +9.4   | 37.5     | 58.5      | 40.4      | 21.5     | 41.5     | 48.3     |
| CARAFE       | R50      | +0.3   | *38.6*   | *59.9*    | *42.2*    | *23.3*   | *42.2*   | *49.7*   |
| IndexNet     | R50      | +8.4   | 37.6     | 58.4      | 40.9      | 21.5     | 41.3     | 49.2     |
| A2U          | R50      | +0.1   | 37.3     | 58.7      | 40.0      | 21.7     | 41.1     | 48.5     |
| FADE         | R50      | +0.2   | 38.5     | 59.6      | 41.8      | 23.1     | *42.2*   | 49.3     |
| SAPA-I       | R50      | +0     | 37.7     | 59.2      | 40.6      | 22.2     | 41.2     | 48.4     |
| SAPA-B       | R50      | +0.1   | 37.8     | 59.2      | 40.6      | 22.4     | 41.4     | 49.1     |
| SAPA-D       | R50      | +0.6   | **39.2** | **60.8**  | **42.7**  | **23.5** | **42.9** | **50.3** |
| Nearest      | R101     | 65.8   | 39.4     | 60.1      | 43.1      | 22.4     | 43.7     | 51.1     |
| SAPA-D       | R101     | +0.6   | **40.6** | **61.8**  | **44.1**  | **24.3** | **45.0** | **52.8** |
