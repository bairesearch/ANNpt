"""ANNpt_ANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt ANN artificial neural network model

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers

class ANNconfig():
	def __init__(self, batchSize, numberOfLayers, numberOfConvlayers, hiddenLayerSize, CNNhiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.numberOfConvlayers = numberOfConvlayers
		self.hiddenLayerSize = hiddenLayerSize
		self.CNNhiddenLayerSize = CNNhiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples

class BiasLayer(nn.Module):
	def __init__(self, num_features):
		super(BiasLayer, self).__init__()
		self.bias = nn.Parameter(pt.zeros(num_features), requires_grad=True)

	def forward(self, x):
		# Adding bias to every element of the input tensor x
		return x + self.bias.unsqueeze(0)
			
class ANNmodel(nn.Module):
		
	def __init__(self, config):
		super().__init__()
		self.config = config
			
		layersLinearListF = []
		layersLinearListF.append(None)	#input layer i=0
		
		self.n_h = self.generateLayerSizeList()
		print("self.n_h = ", self.n_h)
		
		self.batchNorm = None
		self.batchNormFC = None
		self.dropout = None
		self.maxPool = None

		if(useCNNlayers):
			if CNNbatchNorm:
				self.batchNorm = nn.ModuleList()
				for layerIndex in range(config.numberOfConvlayers):
					_, in_channels, out_channels, _, _, _, _, _ = ANNpt_linearSublayers.getCNNproperties(self, layerIndex)
					self.batchNorm.append(nn.BatchNorm2d(out_channels))
			if batchNormFC:
				self.batchNormFC = nn.ModuleList()
				for layerIndex in range(numberOfConvlayers, config.numberOfLayers):
					# determine number of features out of that FC
					if layerIndex == config.numberOfLayers-1:
						feat = config.outputLayerSize
					else:
						feat = config.hiddenLayerSize
					self.batchNormFC.append(nn.BatchNorm1d(feat))
			if(dropout):
				self.dropout = nn.ModuleList()
				for _ in range(config.numberOfLayers):
					self.dropout.append(nn.Dropout2d(p=dropoutProb))
			if(CNNmaxPool):
				self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

		for l2 in range(1, config.numberOfLayers+1):	
			l1 = self.getLayerIndex(l2)
			if(useCNNlayers and l2<=numberOfConvlayers and l2<config.numberOfLayers):	#do not use CNN for final layer
				linearF = ANNpt_linearSublayers.generateLinearLayerCNN(self, l1, config, forward=True, bias=True, layerIndex2=l2)
			else:
				linearF = ANNpt_linearSublayers.generateLinearLayer(self, l1, config, forward=True, bias=True, layerIndex2=l2)
			layersLinearListF.append(linearF)
		self.layersLinearF = nn.ModuleList(layersLinearListF)
		self.activationF = ANNpt_linearSublayers.generateActivationFunction(activationFunctionTypeForward)
		if(useInbuiltCrossEntropyLossFunction):
			self.lossFunctionFinal = nn.CrossEntropyLoss()
		else:
			self.lossFunctionFinal = nn.NLLLoss()	#nn.CrossEntropyLoss == NLLLoss(log(softmax(x)))
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		self.Ztrace = [None]*(config.numberOfLayers+1)
		self.Atrace = [None]*(config.numberOfLayers+1)

	def generateLayerSizeList(self):
		n_h = [None]*(self.config.numberOfLayers+1)
		for l2 in range(0, self.config.numberOfLayers+1):
			if(l2 == 0):
				n_h[l2] = self.config.inputLayerSize
			elif(l2 == numberOfLayers):
				n_h[l2] = self.config.outputLayerSize
			else:
				if(useCNNlayers and l2<=numberOfConvlayers):
					n_h[l2] = self.config.CNNhiddenLayerSize
				else:
					n_h[l2] = self.config.hiddenLayerSize
		return n_h
			
	def forward(self, trainOrTest, x, y, optim, layer=None):
			
		outputPred = None
		outputTarget = None

		batch_size = x.shape[0]
		if(useImageDataset):
			#model code always assumes data dimensions are flattened;
			x = x.reshape(batch_size, -1)

		for l2 in range(1, numberOfLayers+1):
			self.Ztrace[l2] = pt.zeros([batch_size, self.n_h[l2]], device=device)
			self.Atrace[l2] = pt.zeros([batch_size, self.n_h[l2]], device=device)

		outputPred = x #in case layer=0

		AprevLayer = x
		self.Atrace[0] = AprevLayer
		self.Ztrace[0] = pt.zeros_like(AprevLayer)	#set to zero as not used (just used for shape initialisation)
		
		maxLayer = self.config.numberOfLayers
		
		for l2 in range(1, maxLayer+1):
			#print("l2 = ", l2)
			#print("self.config.numberOfLayers = ", self.config.numberOfLayers)
			
			A, Z, inputTarget = self.neuralNetworkPropagationLayerForward(l2, AprevLayer)
			
			if(l2 == self.config.numberOfLayers):
				outputPred = Z	#activation function softmax is applied by self.lossFunctionFinal = nn.CrossEntropyLoss()
			else:
				outputPred = A
			if(l2 == self.config.numberOfLayers):
				outputTarget = y
				if(trainOrTest):
					loss, accuracy = self.trainLayerFinal(l2, outputPred, outputTarget, optim, calculateAccuracy=True)
				else:
					loss, accuracy = self.calculateLossAccuracy(outputPred, outputTarget, self.lossFunctionFinal, calculateAccuracy=True)
			
			AprevLayer = A
			self.Ztrace[l2] = Z
			self.Atrace[l2] = A
				
		return loss, accuracy

	def getLayerIndex(self, l2):
		l1 = l2-1
		return l1
	
	def calculateActivationDerivative(self, A):
		Aactive = (A > 0).float()	#derivative of relu
		return Aactive
	
	def calcMSEDerivative(self, pred, target):
		mse_per_neuron = (pred - target)
		return mse_per_neuron

	def trainLayerFinal(self, l2, pred, target, optim, calculateAccuracy=False):
		lossFunction = self.lossFunctionFinal
		layerIndex = self.getLayerIndex(l2)
		loss, accuracy = self.calculateLossAccuracy(pred, target, lossFunction, calculateAccuracy)
		if(trainLocal):
			opt = optim[layerIndex]
			opt.zero_grad()
			loss.backward()
			opt.step()
		return loss, accuracy
				
	def trainLayerHidden(self, l2, AprevLayer, Ahidden, Itarget, Otarget, optim, calculateAccuracy=False):
		
		Ipred = None
		Opred = None
		
		loss, accuracy = self.trainLayerHiddenBackpropAuto(l2, Ahidden, Itarget, Ipred, optim, Otarget, Opred, calculateAccuracy)

		return loss, accuracy

	def trainLayerHiddenBackpropAuto(self, l2, Ahidden, Itarget, Ipred, optim, Otarget, Opred, calculateAccuracy=False):
		layerIndex = self.getLayerIndex(l2)
		opt = optim[layerIndex]
		opt.zero_grad()
		loss.backward()
		opt.step()	
		
		return loss, accuracy

	def calculateLossAccuracy(self, pred, target, lossFunction, calculateAccuracy=False):
		accuracy = 0
		if(calculateAccuracy):
			accuracy = self.accuracyFunction(pred, target)
		loss = lossFunction(pred, target)
		return loss, accuracy

	def neuralNetworkPropagationLayerForward(self, l2, AprevLayer):

		#print("neuralNetworkPropagationLayerForward l2 = ", l2)
		inputTarget = None

		cnn = False
		if(useCNNlayers and l2<=numberOfConvlayers and l2<self.config.numberOfLayers):	#do not use CNN for final layer
			cnn = True
		Z = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l2), AprevLayer, self.layersLinearF[l2], cnn=cnn)
		maxPoolAfterActivation = True

		Z = ANNpt_linearSublayers.executeResidual(self, self.getLayerIndex(l2), l2, Z, AprevLayer)	#residual
		Z = ANNpt_linearSublayers.convReshapeIntermediate(self, self.getLayerIndex(l2), Z, supportMaxPool=maxPoolAfterActivation)
		Z = ANNpt_linearSublayers.executeBatchNormLayer(self, self.getLayerIndex(l2), Z, self.batchNorm, self.batchNormFC)	#batchNorm
		A = ANNpt_linearSublayers.executeActivationLayer(self, self.getLayerIndex(l2), Z, self.activationF)	#relU
		A = ANNpt_linearSublayers.executeDropoutLayer(self, self.getLayerIndex(l2), A, self.dropout)	#dropout
		if(maxPoolAfterActivation):
			A = ANNpt_linearSublayers.executeMaxPoolLayer(self, self.getLayerIndex(l2), A, self.maxPool)	#maxPool
		A = ANNpt_linearSublayers.convReshapeFinal(self, self.getLayerIndex(l2), A)

		return A, Z, inputTarget



