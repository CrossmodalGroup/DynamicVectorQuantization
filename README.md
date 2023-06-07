# DynamicVectorQuantization (CVPR 2023 highlight)

Offical PyTorch implementation of our CVPR 2023 highlight paper "[Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Towards_Accurate_Image_Coding_Improved_Autoregressive_Image_Generation_With_Dynamic_CVPR_2023_paper.pdf)".

**TL;DR** For vector-quantization (VQ) based autoregressive image generation, we propose a novel *variable-length* coding to replace existing *fixed-length* coding, which brings an accurate & compact code representation for images and a natural *coarse-to-fine* autoregressive generation order. 

Our framework includes: (1) DynamicQuantization VAE (DQ-VAE) which encodes image regions into variable-length codes based on their information densities. (2) DQ-Transformer which thereby generates images autoregressively from coarse-grained (smooth regions with fewer codes) to fine-grained (details regions with
more codes) by modeling the position and content of codes in each granularity alternately, through a novel stackedtransformer architecture and shared-content, non-shared position input layers designs.

**See Our Another CVPR2023 Work about Vector-Quantization based Image Generation**  "[Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Not_All_Image_Regions_Matter_Masked_Vector_Quantization_for_Autoregressive_CVPR_2023_paper.pdf)" ([GitHub](https://github.com/CrossmodalGroup/MaskedVectorQuantization))


![image](assets/dynamic_framework.png)

# Requirements and Installation
Please run the following command to install the necessary dependencies.

```
conda env create -f environment.yml
```

# Data Preparation
Prepare dataset as follows, then change the corresponding datapath in `data/default.py`.

## ImageNet
Prepare ImageNet dataset structure as follows:

```
${Your Data Root Path}/ImageNet/
├── train
│   ├── n01440764
│   |   |── n01440764_10026.JPEG
│   |   |── n01440764_10027.JPEG
│   |   |── ...
│   ├── n01443537
│   |   |── n01443537_2.JPEG
│   |   |── n01443537_16.JPEG
│   |   |── ...
│   ├── ...
├── val
│   ├── n01440764
│   |   |── ILSVRC2012_val_00000293.JPEG
│   |   |── ILSVRC2012_val_00002138.JPEG
│   |   |── ...
│   ├── n01443537
│   |   |── ILSVRC2012_val_00000236.JPEG
│   |   |── ILSVRC2012_val_00000262.JPEG
│   |   |── ...
│   ├── ...
├── imagenet_idx_to_synset.yml
├── synset_human.txt
```

## FFHQ
The FFHQ dataset could be obtained from the [FFHQ repository](https://github.com/NVlabs/ffhq-dataset). Then prepare the dataset structure as follows:
```
${Your Data Root Path}/FFHQ/
├── assets
│   ├── ffhqtrain.txt
│   ├── ffhqvalidation.txt
├── FFHQ
│   ├── 00000.png
│   ├── 00001.png
```

# Training of DQVAE
## Training DQVAE with dual granularities (downsampling factor of F=16 and F=8)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus -1 --base configs/stage1/dqvae-dual-r-05_imagenet.yml --max_epochs 50
```
The target ratio for the finer granularity (F=8) could be set in `model.params.lossconfig.params.budget_loss_config.params.target_ratio`.

## Training DQVAE with triple granularities (downsampling factor of F=32, F=16, F=8)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus -1 --base configs/stage1/dqvae-triple-r-03-03_imagenet.yml --max_epochs 50
```
The target ratio for the finest granularity (F=8) could be set in `model.params.lossconfig.params.budget_loss_config.params.target_fine_ratio`. The target ratio for the median granularity (F=16) could be set in `model.params.lossconfig.params.budget_loss_config.params.target_median_ratio`.

## *A Better Version of DQVAE*
Here we provide a better version of DQVAE compare with the one we proposed in the paper, which leads to much stable training results and also slight better reconstruction quality. To be specific, we assign the granularity of each region directly according to their image entropy instead of the features extracted from the encoder.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus -1 --base configs/stage1/dqvae-entropy-dual-r05_imagenet.yml --max_epochs 50
```
The target ratio for the finer granularity (F=8) could be set in `model.params.encoderconfig.params.router_config.params.fine_grain_ratito`. The distribution of image entropy is pre-calculated in `scripts/tools/thresholds/entropy_thresholds_imagenet_train_patch-16.json`.

## Visualization of Variable-length Coding
![image](assets/dynamic_visual2.png)

# Training of DQ-Transformer

## Unconditional Generation

Copy the first stage model DQ-VAE's config to `model.params.first_stage_config`. The pre-trained DQ-VAE's path should be set in `model.params.first_stage_config.params.ckpt_path`. Here we take ImageNet as an example to show the unconditional DQ-Transformer training, and other datasets like FFHQ could be derive correspondingly.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus -1 --base configs/stage2/uncond_imagenet_p6c18.yml --max_epochs 100
```

NOTE: some important hyper-parameters in the config file: 
- the layer of Content-Transformer: `model.params.transformer_config.params.content_layer`
- the layer of Position-Transformer: `model.params.transformer_config.params.position_layer`
- the vocab size of content code: `model.params.transformer_config.params.vocab_size`, which should include the codebook size of DQ-VAE's codebook, 1 extra pad code, 1 extra eos code and 1 extra sos code.
- the vocab size of coarse granularity's position: `model.params.transformer_config.params.coarse_position_size`, which should include the size of coarse granularity's feature map (e.g., 16 $\times$ 16 = 256 for downsampling factor F=16, or 32 $\times$ 32 = 1024 for downsampling factor F=8), 1 extra pad code, 1 extra eos code and 1 extra sos code.
- the vocab size of fine granularity's position: `model.params.transformer_config.params.fine_position_size`

## Class-conditional Generation

Copy the first stage model DQ-VAE's config to `model.params.first_stage_config`. The pre-trained DQ-VAE's path should be set in `model.params.first_stage_config.params.ckpt_path`. The class-conditional training for DQ-Transformer:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus -1 --base configs/stage2/class_imagenet_p6c18.yml --max_epochs 100
```

NOTE: some important hyper-parameters in the config file: 
- the vocab size of content code: `model.params.transformer_config.params.vocab_size`, which should include the codebook size of DQ-VAE's codebook, 1 extra pad code, 1 extra eos code and 1000 imagenet class number.
- the vocab size of coarse granularity's position: `model.params.transformer_config.params.coarse_position_size`, which should include the size of coarse granularity's feature map (e.g., 16 $\times$ 16 = 256 for downsampling factor F=16, or 32 $\times$ 32 = 1024 for downsampling factor F=8), 1 extra pad code, 1 extra eos code and 1000 imagenet class number.
- the vocab size of fine granularity's position: `model.params.transformer_config.params.fine_position_size`

## Pre-trained Models

| description | Training Details | Dataset | FID (val, 50k) | download link |
| ----------- | ---------------- | --------| -------------- | ------------- |
| DQ-VAE, dual granularity ($F_8=0.5, F_{16}=0.5$) (entropy-based) | 4 A100, 10 epochs | ImageNet | 1.6968 | [Google Cloud](https://drive.google.com/drive/folders/19pK7bfK40FxbZLJTcx8VHX0lqLbm5ECu?usp=sharing) |

## Reference
If you found this code useful, please cite the following paper:

```
@InProceedings{Huang_2023_CVPR,
    author    = {Huang, Mengqi and Mao, Zhendong and Chen, Zhuowei and Zhang, Yongdong},
    title     = {Towards Accurate Image Coding: Improved Autoregressive Image Generation With Dynamic Vector Quantization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22596-22605}
}
```

```
@InProceedings{Huang_2023_CVPR,
    author    = {Huang, Mengqi and Mao, Zhendong and Wang, Quan and Zhang, Yongdong},
    title     = {Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {2002-2011}
}
```