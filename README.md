# HiDe-Prompt
The official implementation of HiDe-Prompt. 

This repository is modified from PyTorch implementation of [Dual-Prompt](https://github.com/JH-LEE-KR/dualprompt-pytorch). 

## Requirements
- Python 3.6+  
```pip install -r requirements.txt```

## Experimental Setup
Our code has been tested on four datasets: CIFAR-100, ImageNetR, 5-datasets, and CUB200:
### dataset
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- 5-datasets (including SVHN, MNIST, CIFAR10, NotMNIST, FashionMNIST)
- [CUB200](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

### supervised and self-supervised checkpoints
and incorporated the following supervised and self-supervised checkpoints as backbones:
- [sup21k vit](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)
- [ibot21k](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint.pth)
- [ibot](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth)
- [mocov3](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar)
- [dino](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth)  
  
Please download the self-supervised checkpoints and put them in the /checkpoints/{checkpoint_name} directory.

## Usage
To reproduce the results mentioned in our paper, execute the training script in /training_script/{train_{dataset}_{backbone}.sh}.

Our code also supports reproducing the original DualPrompt experiments by modifying the hyper-parameters in the /config/{dataset}_dualprompt.py file.

If you encounter any issues or have any questions, please let us know. 