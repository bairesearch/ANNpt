"""ANNpt_ANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt ANN artificial neural network

"""

from ANNpt_globalDefs import *
from torchsummary import summary
import ANNpt_ANNmodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	
	if(printANNmodelProperties):
		print("Creating new model:")
		print("\t ---")
		print("\t datasetType = ", datasetType)
		print("\t stateTrainDataset = ", stateTrainDataset)
		print("\t stateTestDataset = ", stateTestDataset)
		print("\t ---")
		print("\t datasetName = ", datasetName)
		print("\t datasetRepeatSize = ", datasetRepeatSize)
		print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
		print("\t ---")
		print("\t batchSize = ", batchSize)
		print("\t numberOfLayers = ", numberOfLayers)
		print("\t hiddenLayerSize = ", hiddenLayerSize)
		print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
		print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
		print("\t ---")
		print("\t useImageDataset = ", useImageDataset)
		if(useImageDataset):
			print("\t useCNNlayers = ", useCNNlayers)
			print("\t numberOfConvlayers = ", numberOfConvlayers)
			print("\t numberOfFFLayers = ", numberOfFFLayers)
			
	config = ANNpt_ANNmodel.ANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		numberOfConvlayers = numberOfConvlayers,
		hiddenLayerSize = hiddenLayerSize,
		CNNhiddenLayerSize = CNNhiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples
	)
	model = ANNpt_ANNmodel.ANNmodel(config)
	
	print(model)
	#summary(model, input_size=(3, 32, 32))  # adjust input_size as needed

	return model
	
