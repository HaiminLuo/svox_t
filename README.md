# svox_t
Project Page: https://haiminluo.github.io/publication/artemis

This repository contains a differentiable dynamic feature-level octree and renderer implementation
as a PyTorch CUDA extension described in  

Artemis: Articulated Neural Pets with Appearance and Motion Synthesis<br>
Luo, Haimin and Xu, Teng and Jiang, Yuheng and Zhou, Chenglin and Qiu, Qiwei and Zhang, Yingliang and Yang, Wei and Xu, Lan and Yu, Jingyi

This codebase supports real-time volume animating, real-time octree construction and real-time feature-level volume rendering with opacity/depth generation. 

Please also refer to the following repositories

- Artemis - NGI animals and high quality multi-view datasets for furry animals with motions: <https://github.com/HaiminLuo/Artemis>

## Installation
`python setup.py install`

## Misc
`svox_t` stands for **s**parse **v**oxel **o**ctree e**x**tension for **t**emporal scenes.

## Acknowledgement
We would like to thank [PlenOctree](https://github.com/sxyu/plenoctree) authors for releasing their implementations.


## Citation
If you find our code or paper helps, please consider citing:
```
@article{10.1145/3528223.3530086,
author = {Luo, Haimin and Xu, Teng and Jiang, Yuheng and Zhou, Chenglin and Qiu, Qiwei and Zhang, Yingliang and Yang, Wei and Xu, Lan and Yu, Jingyi},
title = {Artemis: Articulated Neural Pets with Appearance and Motion Synthesis},
year = {2022},
issue_date = {July 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3528223.3530086},
doi = {10.1145/3528223.3530086},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {164},
numpages = {19},
keywords = {novel view syntheis, neural rendering, dynamic scene modeling, neural volumetric animal, motion synthesis, neural representation}
}
```