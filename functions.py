import torch as tch
import torchvision.models as md 
import torchvision.transforms as t 

#import cv2 as cv2		# not sure to need it
import numpy as np 
from PIL import Image
np.set_printoptions(threshold=np.inf)



def mask(img, idx):	
	""" creates a mask from a segmented image,
		each pixel from the class we are interested in will be set to 1, the others to 0 """

	shape = (np.shape(img)[0], np.shape(img)[1], 3)
	msk = np.zeros(shape, dtype = np.uint8)

	for i, col in enumerate(img):
		for j, v in enumerate(col):

			if v == idx:
				msk[i, j] = [1, 1, 1]

	return msk

def mulem(a, b):

	""" makes an "element wise" ma"""


def seeyagreenbg(im_name):

	""" Takes a picture with one or several human beings, puts them on another one,
		can work with other classes of the model """

	###### First of all we need a model segment the images, there it is :

	net = md.segmentation.fcn_resnet101(pretrained=True)
	net.eval()### !!!! never forget this !!!!!

	### it got 21 classes : 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
	###						6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=dining table, 12=dog, 13=horse
	###						14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
	### Since we are interested in human beings we gonna use the 15th class


	#### then the preprocessing :

	pp = t.Compose([t.Resize(256), 
                   t.CenterCrop(224), 
                   t.ToTensor(), 
                   t.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])

	pilme = t.Compose([t.ToPILImage()])		#### to watch the final result


	##### open the image, preprocess it and pass it through the network

	img = Image.open(im_name)
	width, height = img.size		#### we'll need it

	prep_img = pp(img).unsqueeze(0)
	net_img = net(prep_img)['out']			##### shape = (1, 21, 224, 224)
	

	### Now we gonna create a mask
	om = tch.argmax(net_img.squeeze(), dim=0).detach().cpu().numpy()
	msk = mask(om, 15)

	##### and apply it to the original image

	#img.resize((224, 224))
	img_np = np.array(img.resize((224, 224)))

	masked = img_np * msk #np.array(list(img_np[i, j]*msk[i, j] for i, j in range(np.shape(msk))))

	i_masked = Image.fromarray(masked)
	i_masked.show()

	