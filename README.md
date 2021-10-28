# Addressing catastrophic forgetting for medical domain expansion

## About

This is the code repository for our paper "Addressing catastrophic forgetting for medical domain expansion", available on preprint here: https://arxiv.org/abs/2103.13511

## Dependencies

To pull the docker container:

docker pull projectmonai/monai:latest

Then install the packages in requirements.txt

## Code

train_model.py: Train a base model on the original domain
train_model_ft_bn.py: Fine-tune batch normalization layers only on a second domain
train_model_ft_bn_freeze.py: Fine-tune batch normalization layers only on a second domain while freezing batch normalization statistics from the original domain
train_model_ft.py: Fine-tune all layers on a second domain 
train_model_ft_freeze.py: Fine-tune all layers on a second domain while freezing batch normalization statistics from the original domain

fim.py: Calculate the fisher matrix from the original domain (adapted from 
train_model_ft_bn_ewc.py: Fine-tune batch normalization layers only with use of Elastic Weight Consolidation on a second domain
train_model_ft_bn_freeze_ewc.py: Fine-tune batch normalization layers with use of Elastic Weight Consolidation only on a second domain while freezing batch normalization statistics from the original domain
train_model_ft_ewc.py: Fine-tune all layers with use of Elastic Weight Consolidation on a second domain 
train_model_ft_freeze_ewc.py: Fine-tune all layers with use of Elastic Weight Consolidation on a second domain while freezing batch normalization statistics from the original domain

run_inference.py: run model inference
