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
import pickle
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
pathFileTrain = #List of Training files
labelTrain = #List of Training labels
#
pathFileValid = #List of Validation files
labelValid = #List of Validation labels
#
#Repetitions
for rep in ['1','2','3','4','5']:
	modelname =  'Name to save model as'+rep
	datadir = 	# Indicate path where you want the model saved
	lossFilePath = 	# Path loss log file
	#
	# Batch size, maximum number of epochs
	trBatchSize = 64
	trMaxEpoch = 50
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
			RandRotate(range_x=15, prob=0.1, keep_size=True), # low probability for rotation 
			RandFlip(spatial_axis=0, prob=0.5),# left right flip 
			RandFlip(spatial_axis=1, prob=0.5), # horizontal flip
			RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
			AsChannelFirst(),
			ToTensor()
		]
	)

	transformSequence_valid = Compose(
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
	datasetValid = My_Dataset(pathFileValid, labelValid, transformSequence_valid)
	print(len(datasetValid))
	dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
	dataLoaderValid = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
	#
	# SET DEVICE 
	deviceNum = 0
	device = torch.device(deviceNum if torch.cuda.is_available() else "cpu")
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
	# Send model to GPU
	model = NetArch().to(device)
	optimizer = optim.Adam(model.parameters(), lr = 1e-4, betas=(0.9, 0.999))
	criterion = torch.nn.CrossEntropyLoss()
	#
	#Open the loss file
	lossFile = open(lossFilePath, "w")
	#
	# Initialize checkpoint counter and best loss
	numCheckpoints = 1
	bestvalloss = 100
	#
	#Train the model
	for epochID in range(0, trMaxEpoch):

		print("Epoch " + str(epochID), end =" ")

		running_loss_train = 0.0
		for batchcount, (varInput, target) in enumerate(dataLoaderTrain):
			print(batchcount, end=" ") 
			model.train()
			inputs = varInput.to(device)
			labels = target.to(device)
			optimizer.zero_grad() 
			with torch.set_grad_enabled(True): # enable gradient while training
				outputs = model(inputs)
				trainloss = criterion(outputs, labels)
				trainloss.backward()
				optimizer.step()

			lossFile.write("Batch " + str(batchcount) + " train loss = " + str(trainloss.item()) + "\n")
			lossFile.flush()
			running_loss_train += trainloss.item()*inputs.size(0)

		# Check validation loss after each epoch
		model.eval() # Evaluation mode
		running_loss_val = 0.0
		for batchcount, (varInput, target) in enumerate(dataLoaderValid):
			print(batchcount, end=" ")
			inputs = varInput.to(device)
			labels = target.to(device)
			with torch.set_grad_enabled(False): # don't change the gradient for validation
				outputs = model(inputs)
				validloss = criterion(outputs, labels)
			running_loss_val += validloss.item()*inputs.size(0)	

		epoch_loss_train = running_loss_train / len(datasetTrain)
		epoch_loss_val = running_loss_val / len(datasetValid)
		print('Train Loss: {:.4f} Val Loss: {:.4f}'.format(epoch_loss_train, epoch_loss_val))

		lossFile.write("Epoch " + str(epochID) + " train loss = " + str(epoch_loss_train) + " valid loss = " + str(epoch_loss_val) + "\n")
		lossFile.flush()

		# Save model w/ lowest loss
		if epoch_loss_val < bestvalloss:
			torch.save(model.state_dict(), datadir+modelname+'.pth.tar')
			bestvalloss = epoch_loss_val

	lossFile.close()
	#
