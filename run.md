﻿
# Usage Guide

------------------

## Prerequisites

The major libraries to run the code are as follow. If the version of PyTorch is not 0.4.1, the experimental results may differ from the results in the paper.

- [PyTorch==0.4.1][pytorch]
- [scikit-learn==0.21.3][sklearn]
- [scipy==1.0.0][scipy]

## Feature extraction

- Calculate vectors of the  Inception-v3-RGB and Inception-v3-Flow features according to the [URL][tsn].
- Calculate vectors of the  VGGish feature according to the [URL][VGGish].

## Data Preparation

Download the feature vectors of VideoEmotion from [Baiduyun][urldata], and the password is ``785q``. Then, extract the downloaded files to the directory ``./data``.

## Testing Provided Model

Download the trained models for VideoEmotion from [Baiduyun][urlmodel] by using the password ``v015``, and copy the downloaded directorys (i.e., ``models`` and ``models-best``) to the source code directory. To evaluate the models trained on VideoEmotion-4 and VideoEmotion-8, run the scripts as follows.

```
./test-VideoEmotion-4.sh
./test-VideoEmotion-8.sh
```

## Training Model

To train the model on VideoEmotion-4 and VideoEmotion-8, run the scripts as follows.

```
./train-val-VideoEmotion-4.sh
./train-val-VideoEmotion-8.sh
```

## Reference

If you find the code useful, please cite the following paper:

```
@article{Yi2019Affective,
  title={Affective Video Content Analysis with Adaptive Fusion Recurrent Network},
  author={Yi, Yun and Wang, Hanli and Li, Qinyu},
  journal={IEEE Transactions on Multimedia}
}
``` 

[pytorch]:https://pytorch.org
[sklearn]:http://scikit-learn.org/stable/index.html
[scipy]:https://www.scipy.org
[tsn]:https://github.com/yjxiong/temporal-segment-networks
[VGGish]:https://github.com/tensorflow/models/tree/master/research/audioset
[urldata]:https://pan.baidu.com/s/1aRFdnJB_bRJymuHKwDNbWQ
[urlmodel]:https://pan.baidu.com/s/1o1hamk0h5mmDO8IqO0V56Q
