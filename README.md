# FFAVOD
Repository for the paper (currently under review) FFAVOD: Feature Fusion Architecture for Video Object Detection
<br> by Hughes Perreault<sup>1</sup>, Guillaume-Alexandre Bilodeau<sup>1</sup>, Nicolas Saunier<sup>1</sup> and Maguelonne Héritier<sup>2</sup>.
<br>
<sup>1</sup> Polytechnique Montréal
<sup>2</sup> Genetec <br>

## Abstract
A significant amount of redundancy exists between consecutive frames of a video. Object detectors typically produce detections for one image at a time, without any capabilities for taking advantage of this redundancy. Meanwhile, many applications for object detection work with videos, including intelligent transportation systems, advanced driver assistance systems and video surveillance. Our work aims at taking advantage of the similarity between video frames to produce better detections. We propose FFAVOD, standing for feature fusion architecture for video object detection. We first introduce a novel video object detection architecture that allows a network to share feature maps between nearby frames. Second, we propose a feature fusion module that learns to merge feature maps to enhance them. We show that using the proposed architecture and the fusion module can improve the performance of three base object detectors on two object detection benchmarks containing sequences of moving road users. Using our architecture on the SpotNet base detector, we obtain the state-of-the-art performance on the UA-DETRAC public benchmark as well as on the UAVDT dataset.


## Model
![Architecture](imgs/architecture.jpg "")

A visual representation of FFAVOD with a window of 5 frames (n=2). Frames are passed through the backbone network of the base object detection network, and the fusion module takes their outputs as input. Finally, the fusion module outputs a fused feature map compatible with the base object detection network, and the base object detection heads are applied to the fused feature map to classify the object categories and regress the bounding boxes.

<img src="https://github.com/hu64/FFAVOD/blob/master/imgs/fusion_module.jpg?raw=true" alt="The Fusion Module"/>

The fusion module. Channels are represented by colors. The fusion module  is  composed  of  channel  grouping,  concatenation  followed  by  1×1 convolution and a final re-ordering of channels.

## Results

For the official references, please refer to the paper.

### Results on UA-DETRAC

| Detector                                                 | Overall          | Easy             | Medium           | Hard             | Cloudy           | Night            | Rainy            | Sunny            |
|----------------------------------------------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| FFAVOD-SpotNet                                           | 88.10% | 97.82% | 92.84% | 79.14% | 91.25% | 89.55% | 82.85% | 91.72%          |
| SpotNet with U-Net | 87.76%          | 97.78%          | 92.57%          | 78.59%          | 90.88%          | 89.28%          | 82.47%          | 91.83% |
| SpotNet                     | 86.80%          | 97.58%          | 92.57%          | 76.58%          | 89.38%          | 89.53%          | 80.93%          | 91.42%          |
| FFAVOD-CenterNet                                        | 86.85% | 97.47% | 92.58% | 76.51% | 89.76% | 89.52% | 80.80% | 90.91% |
| CenterNet             | 83.48%          | 96.50%          | 90.15%          | 71.46%          | 85.01%          | 88.82%          | 77.78%          | 88.73%          |
| FFAVOD-RetinaNet                                        | 70.57% | 87.50% | 75.53% | 58.04% | 80.69% | 69.56% | 56.15% | 83.60% |
| RetinaNet                  | 69.14%          | 86.82%          | 73.70%          | 56.74%          | 79.88%          | 66.57%          | 55.21%          | 82.09%          |
| Joint                           | 83.80% | -                | -                | -                | -                | -                | -                | -                |
| Illuminating               | 80.76%          | 94.56% | 85.90% | 69.72%          | 87.19%          | 80.68% | 71.06% | 89.74%          |
| FG-BR                      | 79.96%          | 93.49%          | 83.60%          | 70.78% | 87.36% | 78.42%          | 70.50%          | 89.8%  |
| HAT                           | 78.64%          | 93.44%          | 83.09%          | 68.04%          | 86.27%          | 78.00%          | 67.97%          | 88.78%          |
| GP-FRCNNm                      | 77.96%          | 92.74%          | 82.39%          | 67.22%          | 83.23%          | 77.75%          | 70.17%          | 86.56%          |
| MMA                                   | 74.88%          | -                | -                | -                | -                | -                | -                | -                |
| Global                            | 74.04%          | 91.57%          | 81.45%          | 59.43%          | -                | 78.50%          | 65.38%          | 83.53%          |
| R-FCN                         | 69.87%          | 93.32%          | 75.67%          | 54.31%          | 74.38%          | 75.09%          | 56.21%          | 84.08%          |
| Perceiving Motion             | 69.10%          | 90.49%          | 75.21%          | 53.53%          | 83.66%          | 73.97%          | 56.11%          | 72.15%          |
| EB                           | 67.96%          | 89.65%          | 73.12%          | 53.64%          | 72.42%          | 73.93%          | 53.40%          | 83.73%          |
| Faster R-CNN                       | 58.45%          | 82.75%          | 63.05%          | 44.25%          | 66.29%          | 69.85%          | 45.16%          | 62.34%          |
| YOLOv2                    | 57.72%          | 83.28%          | 62.25%          | 42.44%          | 57.97%          | 64.53%          | 47.84%          | 69.75%          |
| RN-D                          | 54.69%          | 80.98%          | 59.13%          | 39.23%          | 59.88%          | 54.62%          | 41.11%          | 77.53%          |
| 3D-DETnet                    | 53.30%          | 66.66%          | 59.26%          | 43.22%          | 63.30%          | 52.90%          | 44.27%          | 71.26%          |

### Results on UAVDT

| Detector                                                 | Overall          |
|----------------------------------------------------------|------------------|
| FFAVOD-SpotNet                                          | 53.76% |
| SpotNet with U-Net | 53.38%          |
| SpotNet            | 52.80%          |
| FFAVOD-CenterNet                                        | 52.07% |
| CenterNet       | 51.18%          |
| FFAVOD-RetinaNet                                        | 39.43% |
| RetinaNet                   | 38.26%          |
| LRF-NET                         | 37.81% |
| R-FCN                         | 34.35%          |
| SSD                                   | 33.62%          |
| Faster-RCNN                        | 22.32%          |
| RON                                  | 21.59%          |



## Model Zoo

https://polymtlca0-my.sharepoint.com/:f:/g/personal/hughes_perreault_polymtl_ca/EgdXk5gp0hVMj9D_EaZDgzUBNHCOjeyv1YesZEUYqRP3Wg?e=kpPuXv

## Reproduce results

To reproduce the results, please use the scripts in /ffavod-experiments and adapt the paths. 

## Specific Model Architectures
A visual representation of FFAVOD with a window of five frames (n=2) for each of the three models used.

### RN-VID:
<img src="https://github.com/hu64/FFAVOD/blob/master/imgs/architecture_RN-VID.jpg?raw=true" width="800" alt=""/>

### CenterNet:
<img src="https://github.com/hu64/FFAVOD/blob/master/imgs/architecture_CenterNet.jpg?raw=true" width="800" alt=""/>

### SpotNet:
<img src="https://github.com/hu64/FFAVOD/blob/master/imgs/architecture_SpotNet.jpg?raw=true" width="800" alt=""/>

## Acknowledgements

The code for this paper is mainly built upon [CenterNet](https://github.com/xingyizhou/CenterNet), we would therefore like to thank the authors for providing the source code of their paper. We also acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), [RDCPJ 508883 - 17], and the support of Genetec.

## License

FFAVOD is released under the MIT License. Portions of the code are borrowed from [CenterNet](https://github.com/xingyizhou/CenterNet), [CornerNet](https://github.com/princeton-vl/CornerNet) (hourglassnet, loss functions), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), and [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) (cityscapes dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).
