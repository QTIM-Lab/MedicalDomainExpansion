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
from scipy.special import softmax
#
datadir = #data directory
os.chdir(datadir)
#
models = #List of model names you want to run inference with
#
pathFileTest = #List of Testing files
labelTest = #List of Testing labels
#
# Batch size
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
transformSequence_test = Compose(
	[
		ScaleIntensity(minv=0.0,maxv=1.0),
		AsChannelFirst(),
		ToTensor()
	]
)
#
datasetTest = My_Dataset(pathFileTest, labelTest, transformSequence_test)
print(len(datasetTest))
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
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

for idx,modelname in enumerate(models):
	print(idx,modelname)
	#
	# Indicate path where model is saved
	PATH = datadir+modelname+'.pth.tar'
	#
	model = NetArch()
	model.load_state_dict(torch.load(PATH))
	model.to(device)
	#
	all_outputs = []
	model.eval()
	for batchcount, (varInput, target) in enumerate(dataLoaderTest):
		print(batchcount, end=" ")
		inputs = varInput.to(device)
		with torch.set_grad_enabled(False): # don't change the gradient
			outputs = model(inputs)
		all_outputs = all_outputs + list(np.squeeze(np.array(outputs.cpu())))
	#
	labelTest = np.array(labelTest)
	all_outputs = np.array(all_outputs)
	all_outputs_sm = softmax(all_outputs,axis=1)
	#
	df = pd.DataFrame({'image': pathFileTest, 'label': labelTest, 'output class 0': all_outputs_sm[:,0],'output class 1': all_outputs_sm[:,1]})
	os.chdir('save directory')
	df.to_excel(modelname+'.xlsx')