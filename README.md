# GTOT-Tuning
This is a Pytorch implementation of the following paper:

IJCAI22-[Fine-Tuning Graph Neural Networks via Graph Topology induced Optimal Transport](https://arxiv.org/abs/2203.10453)



If you make use of the code/experiment in your work, please cite our paper (Bibtex below).
```
@article{zhang2022fine,
  title={Fine-tuning graph neural networks via graph topology induced optimal transport},
  author={Zhang, Jiying and Xiao, Xi and Huang, Long-Kai and Rong, Yu and Bian, Yatao},
  journal={arXiv preprint arXiv:2203.10453},
  year={2022}
}
```
<!--
or
```
@inproceedings{zhang2022fine,
  title={Fine-tuning graph neural networks via graph topology induced optimal transport},
  author={Zhang, Jiying and Xiao, Xi and Huang, Long-Kai and Rong, Yu and Bian, Yatao},
  booktitle={the 31st International Joint Conference on Artificial Intelligence and the 25th European Conference on Artificial Intelligence (IJCAI-ECAI 22)},
  page={}
  year={2022}
}
```
-->

![Framework](GTOT-framework.png)


We would like to appreciate the excellent work of [Pretrain-GNNs](https://github.com/snap-stanford/pretrain-gnns) and [TransferLearningLibrary](https://github.com/thuml/Transfer-Learning-Library), which both 
lay a solid foundation for our work.

## Installation
We used the following Python packages for core development. We tested on `Python 3.7.6`.
```
pytorch                   1.4.0           
torch-geometric           1.6.0
torch-scatter             2.0.4 
torch-sparse              0.6.1
torch-spline-conv         1.2.0
rdkit                     2020.03.3.0
tqdm                      4.42.1
tensorboardx              2.1
```


## Dataset download
All the necessary data files can be downloaded from the following links.

For the chemistry dataset, download from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `chem/dataset`.


## Train and test
#### Fine-tuning with GTOT Regularization 
```
sh chem/run.sh
```

#### Saved pre-trained models
The pre-trained models are under `model_gin/` and `model_architecture/`, copied from [Pretrain-GNNs](https://github.com/snap-stanford/pretrain-gnns) .



