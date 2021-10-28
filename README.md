# Addressing catastrophic forgetting for medical domain expansion

## About

This is the code repository for our paper "Addressing catastrophic forgetting for medical domain expansion", available on preprint here: https://arxiv.org/abs/2103.13511

## Installation

1. Install Docker from Docker's website here: https://www.docker.com/get-started. Follow instructions on that link to get Docker set up properly on your workstation.

2. Install the Docker Engine Utility for NVIDIA GPUs, AKA nvidia-docker. You can find installation instructions at their Github page, here: https://github.com/NVIDIA/nvidia-docker

3. Pull docker container using this command: docker pull projectmonai/monai:latest

4. Install the packages in requirements.txt

## Code

**train_model.py**: Train a base model on the original domain
**train_model_ft_bn.py**: Fine-tune batch normalization layers only on a second domain
**train_model_ft_bn_freeze.py**: Fine-tune batch normalization layers only on a second domain while freezing batch normalization statistics from the original domain
**train_model_ft.py**: Fine-tune all layers on a second domain 
**train_model_ft_freeze.py**: Fine-tune all layers on a second domain while freezing batch normalization statistics from the original domain

**fim.py**: Calculate the fisher matrix from the original domain
**train_model_ft_bn_ewc.py**: Fine-tune batch normalization layers only with use of Elastic Weight Consolidation on a second domain
**train_model_ft_bn_freeze_ewc.py**: Fine-tune batch normalization layers with use of Elastic Weight Consolidation only on a second domain while freezing batch normalization statistics from the original domain
**train_model_ft_ewc.py**: Fine-tune all layers with use of Elastic Weight Consolidation on a second domain 
**train_model_ft_freeze_ewc.py**: Fine-tune all layers with use of Elastic Weight Consolidation on a second domain while freezing batch normalization statistics from the original domain

run_inference.py: run model inference

## Contact

This code is under active development, and you may run into errors or want additional features. Send any questions or requests for methods to qtimlab@gmail.com. You can also submit a Github issue if you run into a bug.

## Acknowledgements

Code for fisher matrix calculation adapted from 
