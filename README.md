# DuDoRecon
This repository provides the implementation of our paper "Dual-Domain Multi-Contrast MRI Reconstruction with Synthesis-based Fusion Network".


# Installation
The implementation is developed based on the following Python packages/versions, including:
```
python: 3.8.12
torch: 2.0.0
numpy: 1.20.3
tensorflow: 2.5.0
sigpy: 0.1.23
scipy: 1.10.1
```

# Running Code
The implementation includes three steps: (1) Pre-training the synthesis network to produce the input for registration network, (2) Pre-training the registration network to align the syntheised image with under-sampled target-contrast image as the input for reconstruction network, and (3) Training the reconstruction network for multi-contrast MRI reconstruction. The operations can be performed in both domains.

Only the file `main.py` needs to be executed for running experiments. For example, to train the framework and evaluate the performance with a set of pre-determined hyper-parameters, the command as follows can be executed:
```
python main.py --domain 2 --alpha 0.5 --k_weight 100 --qmodal T2 --un_rate 4 --train_epoch 100
```
In the example above, the framework is optimised using data in both domains, with the regularisation strength of 0.5, k-space loss terms are divided by 100, T2w considered as the target contrast, 4-fold accleration, and training for 100 epochs. 

## Training with Baselines
For training with baseline configurations, two types of baselines are defined. The first one is reconstruction without synthesis-based fusion strategy, which can be defined by specifying the argument `recon_only` as `True`, and whether reference contrast needs to be involved can be defined by specifying the argument `recon_modal`, in which `t2u` and `t1t2u` can be provided for single-contrast and multi-contrast reconstruction, respectively.

## Evaluation
Due to the sequential nature of our proposed framework, the evaluation is integrated inside the training script, in which the evaluation is performed right after training each component. To avoid repeated training, the corresponding lines can be commented for evaluation only in `main.py`, such as line 88, 92, and 96.

## Citation
If this implementation is helpful for your research, please consider citing our work:
