import os
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image       # PIL is currently being developed as Pillow

from transformations import Transformations

class CelebDataSet(Dataset):
    """CelebA dataset
    
    Parameters:
        data_path (str)     -- CelebA dataset main directory(inculduing '/Img' and '/Anno') path
        state (str)         -- dataset phase 'train' | 'val' | 'test'

    Center crop the alingned celeb dataset to 178x178 to include the face area and then downsample to 128x128(Step3).
    In addition, for progressive training, the target image for each step is resized to 32x32(Step1) and 64x64(Step2).
    Orignal images are shaped 218x178 RGB

    Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data 
    available for training models, without actually collecting new data. Data augmentation techniques such as 
    cropping, padding, and horizontal flipping are commonly used to train large neural networks.
    """

    def __init__(self, data_path='./dataset/', state='train', data_aug=False):
        
        ## DEFINE PATHS
        self.main_path = data_path
        self.state = state
        self.data_aug = data_aug

        self.img_path = os.path.join(self.main_path, 'HR')
        self.image_list = os.listdir(self.img_path)
        self.image_list.sort()

        self.transf = Transformations(self.data_aug)


    def __getitem__(self, index):
        """ https://stackoverflow.com/questions/43627405/understanding-getitem-method

        this overwrites the built-in __getitem__ method
        """

        image_path = os.path.join(self.img_path, self.image_list[index])
        x2, x4, target_image, input_image = self.transf.perform(image_path)
        
        return x2, x4, target_image, input_image

    def __len__(self):
        return len(self.image_list)
