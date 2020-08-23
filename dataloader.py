import os
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image       # PIL is currently being developed as Pillow

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

    def __init__(self, data_path='./dataset/', state='train', data_aug=None):
        
        ## DEFINE PATHS
        self.main_path = data_path
        self.state = state
        self.data_aug = data_aug

        self.img_path = os.path.join(self.main_path, 'HR')
        self.image_list = os.listdir(self.img_path)
        self.image_list.sort()

        ## DEFINE totensor OPERATION. FIRST MAKE IT TENSOR (as usual), THEN NORMALIZE
        self.totensor = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

        ## DEFINE THREE SCALING FOR PROGRESSIVE TRAINING
        self._64x64_down_sampling = transforms.Resize((64, 64))
        self._32x32_down_sampling = transforms.Resize((32, 32))
        self._16x16_down_sampling = transforms.Resize((16,16))

    # https://stackoverflow.com/questions/43627405/understanding-getitem-method
    # this overwrites the built-in __getitem__ method
    # for this to take effect: dataloader = CelebDataSet() \ dataloader[i]
    def __getitem__(self, index):
        image_path = os.path.join(self.img_path, self.image_list[index])
        
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

    def __len__(self):
        return len(self.image_list)
