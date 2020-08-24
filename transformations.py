import torchvision.transforms as transforms
import pandas as pd
from os.path import join
from PIL import Image	   # PIL is currently being developed as Pillow

class Transformations:
	def __init__(self, data_aug):

		self.data_aug = data_aug

		## DEFINE PRE-PROCESSING STRATEGY
		if self.data_aug == False:
			self.pre_process = transforms.Compose([
								transforms.CenterCrop((178, 178)),
								transforms.Resize((128,128)),
								])
		else:
			self.pre_process = transforms.Compose([
					transforms.RandomHorizontalFlip(),
					transforms.CenterCrop((178, 178)),
					transforms.Resize((128, 128)),
					transforms.RandomRotation(20, resample=Image.BILINEAR),
					transforms.ColorJitter(brightness=0.4, 
											contrast=0.4, 
											saturation=0.4, 
											hue=0.1)
					])

		self.totensor = transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
							])

		## DEFINE THREE SCALING FOR PROGRESSIVE TRAINING
		self._64x64_down_sampling = transforms.Resize((64, 64))
		self._32x32_down_sampling = transforms.Resize((32, 32))
		self._16x16_down_sampling = transforms.Resize((16,16))


	def perform(self, image_path):
		
		# this shall be your benchmark
		target_image = Image.open(image_path).convert('RGB')
		
		# original downsampled at 64x64. This'll be compared with x2_target_image 2x upsampling
		x4_target_image = self._64x64_down_sampling(target_image)
		
		# original downsampled at 32x32. This'll be compared with input_image 2x upsampling
		x2_target_image = self._32x32_down_sampling(x4_target_image)
		
		# input image is the orginal, downsampled at 16x16
		input_image = self._16x16_down_sampling(x2_target_image)

		# normalize all images
		x2_target_image = self.totensor(x2_target_image)
		x4_target_image = self.totensor(x4_target_image)
		target_image = self.totensor(target_image)
		input_image = self.totensor(input_image)

		return x2_target_image, x4_target_image, target_image, input_image