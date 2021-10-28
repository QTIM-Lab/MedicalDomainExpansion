from __future__ import print_function
import os
import numpy as np
import time
import sys
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
import sklearn.metrics as metrics
import random
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from PIL import Image
import time
import sys
from typing import Dict
from argparse import Namespace
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.utils.data import DataLoader
import torchvision
import cv2
from glob import glob
from monai.transforms import (
	AddChannel,
	AsChannelFirst,
	Compose,
	RandFlip,
	RandRotate,
	RandZoom,
	ScaleIntensity, Transpose, 
	LoadImage,
	ToTensor,
	Resize,
)
from torchvision.transforms import Normalize
#
assert int(torch.__version__.split(".")[1]) >= 4, 'PyTorch 0.4+ required'
#
def fim_diag(model: Module,
			 data_loader: DataLoader,
			 samples_no: int = None,
			 empirical: bool = False,
			 device: torch.device = None,
			 verbose: bool = False,
			 model_name: str = None,
			 every_n: int = None) -> Dict[int, Dict[str, Tensor]]:
	fim = {}
	for name, param in model.named_parameters():
		if param.requires_grad:
			fim[name] = torch.zeros_like(param)

	seen_no = 0
	last = 0
	tic = time.time()
	all_fims = dict({})

	while samples_no is None or seen_no < samples_no:
		data_iterator = iter(data_loader)
		try:
			data, target = next(data_iterator)
		except StopIteration:
			if samples_no is None:
				break
			data_iterator = iter(data_loader)
			data, target = next(data_loader)

		if device is not None:
			data = data.to(device)
			if empirical:
				target = target.to(device)

		logits = model(data)
		if empirical:
			outdx = target.unsqueeze(1)
			# print("emperical outdx is ", outdx)
		else:
			outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
			# print("observed outdx is ", outdx)

		samples = logits.gather(1, outdx)
		# print(samples)

		idx, batch_size = 0, data.size(0)
		while idx < batch_size and (samples_no is None or seen_no < samples_no):
			model.zero_grad()
			torch.autograd.backward(samples[idx], retain_graph=True)
			for name, param in model.named_parameters():
				if param.requires_grad:
					fim[name] += (param.grad * param.grad)
					fim[name].detach_()
			seen_no += 1
			idx += 1

			if verbose and seen_no % 100 == 0:
				toc = time.time()
				fps = float(seen_no - last) / (toc - tic)
				tic, last = toc, seen_no
				sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

			if every_n and seen_no % every_n == 0:
				print("hello")
				all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
									 for (n, f) in fim.items()}

	if verbose:
		if seen_no > last:
			toc = time.time()
			fps = float(seen_no - last) / (toc - tic)
		sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

	all_fims[seen_no] = fim
	print(fim)
	print("Fisher Matrix found !")

	# print(all_fims)
	torch.save(fim, 'Path to save Fisher Matrix')
	return fim
#
pathFileTrain = #List of Training files
labelTrain = #List of Training labels
#
#Repetitions
for rep in ['1','2','3','4','5']:
	#
	deviceNum = 0
	device = torch.device(deviceNum if torch.cuda.is_available() else "cpu")
	#
	trBatchSize = 64
	#
	class My_Dataset(Dataset):
		def __init__(self, image_files, labels, transforms):
			self.image_files = image_files
			self.labels = labels
			self.transforms = transforms
			self.inputs_dtype = torch.float32

		def __len__(self):
			return len(self.image_files)

		def __getitem__(self, index):
			try:
				x_img = cv2.imread(self.image_files[index])
				x_img = cv2.resize(x_img, (224, 224))
				return self.transforms(x_img).type(self.inputs_dtype), self.labels[index]
			except:
				print('Error')
				return None
	#
	#remove "nones" from the batches, ie skipped files
	def collate_fn(batch):
		batch = list(filter(lambda x: x is not None, batch))
		return torch.utils.data.dataloader.default_collate(batch)
	#
	transformSequence_train = Compose(
		[
			ScaleIntensity(minv=0.0,maxv=1.0),
			AsChannelFirst(),
			ToTensor()
		]
	)
	#
	# LOAD DATASET
	datasetTrain = My_Dataset(pathFileTrain, labelTrain, transformSequence_train)
	print(len(datasetTrain))
	dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
	#
	class NetArch(nn.Module):
		def __init__(self):
			super(NetArch, self).__init__()
			self.netarch = torchvision.models.resnet18(pretrained=True)
			num_ftrs = self.netarch.fc.in_features
			self.netarch.fc = nn.Sequential(
				nn.Linear(num_ftrs, 2))

		def forward(self, x):
			x = self.netarch(x)
			return x
	#
	datadir = #Data directory
	modelname =  #Name of model
	PATH = datadir+modelname+'.pth.tar'
	#
	# Send model to GPU
	base_model = NetArch()
	base_model.load_state_dict(torch.load(PATH))	
	base_model.to(device)
	#
	fim_diag(base_model, dataLoaderTrain, empirical=True, samples_no=len(datasetTrain), device= device, verbose=True, model_name = modelname)
	#
