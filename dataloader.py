"""Compare identities

This script first extract one person for each identity. These will be later used to perform super-resolution.
Then, it removes that person from the original datasets.
The super-resolved image will be compared with a face recognition algorithm to understand which group it
belongs to.
"""

import torch
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torchvision import utils
from os.path import join
from PIL import Image       # PIL is currently being developed as Pillow
import argparse
import os
import sys
from math import log10
from ssim import ssim, msssim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd

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

    def __init__(self, data_path = './dataset/'):
        ## DEFINE PATHS
        self.main_path = data_path

        self.img_path = join(self.main_path, 'CelebA/Img/img_align_celeba')
        self.identity_partition_path = join(self.main_path, 'Anno/identity_CelebA.txt')
        self.eval_partition_path = join(self.main_path, 'Anno/list_eval_partition.txt')
        self.final_csv_path = join(self.main_path, 'Anno/definitive.csv')

        # open files and create csv
        # https://stackoverflow.com/a/19008279
        with open(self.eval_partition_path, 'r') as f1, \
             open(self.identity_partition_path, 'r') as f2, \
             open(self.final_csv_path, 'w') as f3:
             for x, y in zip(f1, f2):
                x = x.split()
                y = y.split()
                new_line = '{},{},{}\n'.format(x[0], x[1], y[1])
                f3.write(new_line)  
        
        # https://stackoverflow.com/a/28163238/7347566
        with open(self.final_csv_path, newline='') as fc1:
            r = csv.reader(fc1)
            line = [line for line in r]

        with open(self.final_csv_path, 'w', newline='') as fc2:
            w = csv.writer(fc2)
            w.writerow(['person','group','identity'])
            w.writerows(line)

        ## READ CSV AND EXTRACT VALUES
        df = pd.read_csv(self.final_csv_path)
        
        # extract one person for each identity
        df_sing = df.groupby(['identity']).max()                # one person
        df_sing.reset_index(inplace=True)                       # reset the index
        df_sing = df_sing[['person', 'group', 'identity']]      # reorder to be consistent

        # remove person if already in the previous group
        # https://stackoverflow.com/questions/50449088/check-if-value-from-one-dataframe-exists-in-another-dataframe
        filt = (df['person'].isin(df_sing['person']) == False)
        df_compare = df.loc[filt]

        self.pre_process = transforms.Compose([
                                        transforms.CenterCrop((178, 178)),
                                        transforms.Resize((128,128)),
                                        ])

        self.totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        ## DEFINE THREE SCALING FOR PROGRESSIVE TRAINING
        self._64x64_down_sampling = transforms.Resize((64, 64))
        self._32x32_down_sampling = transforms.Resize((32, 32))
        self._16x16_down_sampling = transforms.Resize((16,16))

        ## RETURN VALUES
        self.df_single = df_sing
        self.df_compare = df_compare

    
    # provide as input an identity number
    def getPerson(self, iden):
        # https://stackoverflow.com/a/47917648
        filt = (self.df_single['identity'] == iden)
        image_path = join(self.img_path, self.df_single.loc[filt]['person'].tolist()[0])
        
        # this shall be your benchmark
        target_image = Image.open(image_path).convert('RGB')
        target_image = self.pre_process(target_image)
        
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

        # debug
        #print('considered image', image_path)

        return x2_target_image, x4_target_image, target_image, input_image


    def getPeople(self):
        return self.df_compare

# ## DEBUG MAIN
# if __name__ == '__main__':
#     dataset = CelebDataSet(data_path='./dataset')
    
#     # dizFull, dizTrain, dizTest = dataset.getDict()
#     x2_target_image, x4_target_image, target_image, input_image = dataset.getPerson(1)

#     # print(df_train)
#     # print(dizTest)