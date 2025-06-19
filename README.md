# CRAS

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/center-aware-residual-anomaly-synthesis-for/multi-class-anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/multi-class-anomaly-detection-on-mvtec-ad?p=center-aware-residual-anomaly-synthesis-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/center-aware-residual-anomaly-synthesis-for/anomaly-detection-on-mpdd)](https://paperswithcode.com/sota/anomaly-detection-on-mpdd?p=center-aware-residual-anomaly-synthesis-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/center-aware-residual-anomaly-synthesis-for/multi-class-anomaly-detection-on-itdd)](https://paperswithcode.com/sota/multi-class-anomaly-detection-on-itdd?p=center-aware-residual-anomaly-synthesis-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/center-aware-residual-anomaly-synthesis-for/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=center-aware-residual-anomaly-synthesis-for)

![](figures/CRAS_schematic.png)

**Center-aware Residual Anomaly Synthesis for Multi-class Industrial Anomaly Detection**

_Qiyu Chen, Huiyuan Luo, Haiming Yao, Wei Luo, Zhen Qu, Chengkan Lv*, Zhengtao Zhang_

[IEEE DOI Link](https://ieeexplore.ieee.org/document/11040032) &
[ArXiv Preprint Link](https://arxiv.org/abs/2505.17551)

## Table of Contents
* [📖 Introduction](#introduction)
* [🔧 Environments](#environments)
* [📊 Data Preparation](#data-preparation)
* [🚀 Run Experiments](#run-experiments)
* [📂 Dataset Release](#dataset-release)
* [🔗 Citation](#citation)
* [🙏 Acknowledgements](#acknowledgements)
* [📜 License](#license)

## Introduction
This repository contains source code for CRAS implemented with PyTorch.
CRAS aims to address inter-class interference and intra-class overlap in multi-class anomaly detection
through center-aware residual learning and distance-guided anomaly synthesis.

This repository also contains the self-built dataset ITDD proposed in our paper.
Here, we present a brief summary of CRAS's performance across several benchmark datasets.

| Multi-class  | MVTec AD |  VisA   |  MPDD   |  ITDD   |
|:------------:|:--------:|:-------:|:-------:|:-------:|
|   I-AUROC    |  98.3%   |  93.5%  |  95.0%  |  99.4%  |
|   P-AUROC    |  98.0%   |  97.7%  |  98.3%  |  97.8%  |

| Single-class | MVTec AD |  VisA   |  MPDD   |  ITDD   |
|:------------:|:--------:|:-------:|:-------:|:-------:|
|   I-AUROC    |  99.7%   |  97.0%  |  98.8%  |  99.6%  |
|   P-AUROC    |  98.4%   |  98.4%  |  98.7%  |  98.0%  |

## Environments
Create a new conda environment and install required packages.
```
conda create -n cras_env python=3.9.21
conda activate cras_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 4090 (24GB).
Same GPU and package version are recommended. 

## Data Preparation
The public datasets employed in the paper are listed below.
The MVTec AD and MPDD datasets retain their original directory structures.
However, you need to run the [provided script](https://github.com/amazon-science/spot-diff/?tab=readme-ov-file#data-preparation)
to convert the VisA dataset to the same layout as MVTec AD.

- MVTec AD ([Download link](https://www.mvtec.com/company/research/datasets/mvtec-ad/))
- MPDD ([Download link](https://github.com/stepanje/MPDD/))
- VisA ([Download link](https://github.com/amazon-science/spot-diff/))

We also release the model weights of CRAS on MVTec AD reported in the paper.
If you want to further explore based on these models,
you need to download the results folder
([Download link](https://drive.google.com/drive/folders/1v6SZD6O3LKVTTliVgp8DeFEqyZ0QCets?usp=sharing))
and move it to the root directory of this repository.
Note that you should clear the existing results folder before moving.

## Run Experiments
Edit `./shell/run-dataset-setting.sh` to configure arguments `--datapath`, `--classes`, and hyperparameter settings.
Please modify argument `--test` to 'ckpt' / 'test' to toggle between training and testing modes.

```
bash run-dataset-setting.sh
```

_"Note that 'dataset' refers to any dataset.
Currently, the `shell` folder contains scripts for four datasets under the single-class and multi-class settings,
such as `run-mvtec-multi.sh` for MVTec AD under multi-class setting. If you want to train your own dataset,
please create a new `run-*-*.sh` file."_

## Dataset Release
### ITDD ([Download link](https://drive.google.com/file/d/1Iy-f_jxJFhSxDc4J0f85wwQKuwj1NzvX/view?usp=sharing/))
The Industrial Textile Defect Detection (ITDD) dataset includes 1885 industrial textile images categorized into 4 categories:
cotton fabric, dyed fabric, hemp fabric, and plaid fabric.
These classes are collected from the industrial production sites of [WEIQIAO Textile](http://www.wqfz.com/en/).
ITDD is an upgraded version of [WFDD](https://github.com/cqylunlun/GLASS?tab=readme-ov-file#1wfdd-download-link)
that reorganizes three original classes and adds one new class.

![](figures/ITDD_samples.png)

## Citation
Please cite the following paper if the code and dataset help your project:

```bibtex
@article{chen2025center,
  title={Center-aware Residual Anomaly Synthesis for Multiclass Industrial Anomaly Detection},
  author={Chen, Qiyu and Luo, Huiyuan and Yao, Haiming and Luo, Wei and Qu, Zhen and Lv, Chengkan and Zhang, Zhengtao},
  journal={IEEE Transactions on Industrial Informatics}, 
  year={2025},
  volume={},
  number={},
  pages={1-11}
}
```

## Acknowledgements
Thanks for the great inspiration from [SimpleNet](https://github.com/DonaldRR/SimpleNet/), [GLASS](https://github.com/cqylunlun/GLASS/),
and [PBAS](https://github.com/cqylunlun/PBAS/).

## License
The code and dataset in this repository are licensed under the [MIT license](https://github.com/cqylunlun/CRAS?tab=MIT-1-ov-file/).
