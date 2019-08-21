import torch as t 
import torchvision.models as md 
import torchvision.transforms as t 


import cv2 as cv2		# not sure to need it
import numpy as np 
from PIL import Image






def seeyagreenbg():

	""" Takes a picture with one or several human beings, puts them on another one,
		can work with other classes of the model """

	###### First of all we need a model segment the images, there it is :

	net = m.segmentation.deeplabv3_resnet101(pretrained=True)

	### it got 21 classes : 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
	###						6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=dining table, 12=dog, 13=horse
	###						14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
	### Since we are interested in human beings we gonna use the 15th class


	#### then the preprocessing :

	pp = t.Compose([T.Resize(256), 
                   T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])


	######### <==== To be continued ....