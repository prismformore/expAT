# expAT: Bi-directional Exponential Angular Triplet Loss for RGB-Infrared Person Re-Identification
![issue in triplet](https://github.com/prismformore/expAT/blob/master/triplet_issue.png)

[arXiv page](https://arxiv.org/abs/2006.00878)

Most existing works use Euclidean metric based constraints to resolve the discrepancy between features of different modalities. However, these methods are incapable of learning angularly discriminative feature embedding because Euclidean distance cannot measure the included angle between embedding vectors effectively. 

As an angularly discriminative feature space is important for classifying the human images based on their embedding vectors, in this paper, we propose a novel ranking loss function, named Bi-directional Exponential Angular Triplet Loss, to help learn an angularly separable common feature space by explicitly constraining the included angles between embedding vectors. 

Moreover, to help stabilize and learn the magnitudes of embedding vectors, we adopt a common space batch normalization layer. 

Quantitative experiments on the SYSU-MM01 and RegDB dataset support our analysis. On SYSU-MM01 dataset, the performance is improved from 7.40% / 11.46% to 38.57% / 38.61% for rank1 accuracy / mAP compared with the baseline. The proposed method can be generalized to the task of single-modality Re-ID and improves the rank-1 accuracy / mAP from 92.0% / 81.7% to 94.7% / 86.6% on the Market-1501 dataset, from 82.6% / 70.6% to 87.6% / 77.1% on the DukeMTMC-reID dataset.

First submitted: April 2019

## Prerequisite
- Python>=3.6
- Pytorch>=1.0.0
- Opencv>=3.1.0
- tensorboard-pytorch

## Run the Code
- Training: ```python main.py -a train``` 
- Testing:  ```python main.py -a test -m checkpoint_name -s test_setting```

# Cite
Please kindly consider citing our paper:
```
@misc{expat,
    title={Bi-directional Exponential Angular Triplet Loss for RGB-Infrared Person Re-Identification},
    author={Hanrong Ye and Hong Liu and Fanyang Meng and Xia Li},
    year={2020},
    eprint={2006.00878},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# Contact:
Hanrong Ye leoyhr AT pku.edu.cn
