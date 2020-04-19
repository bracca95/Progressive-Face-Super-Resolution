from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from os.path import join
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

    def __init__(self, data_path = './dataset/', state = 'train', data_augmentation=None):
        ## DEFINE PATHS
        self.main_path = data_path
        self.state = state
        self.data_augmentation = data_augmentation

        self.img_path = join(self.main_path, 'CelebA/Img/img_align_celeba')
        self.eval_partition_path = join(self.main_path, 'Anno/list_eval_partition.txt')

        ## INIT IMG LISTS
        train_img_list = []
        val_img_list = []
        test_img_list = []

        ## READ FILE list_eval_partition.txt
        f = open(self.eval_partition_path, mode='r')

        while True:
            # the split is used to separate filename from label
            line = f.readline().split()
            if not line: break

            # 0/1/else defines train/val/test
            if line[1] == '0':
                train_img_list.append(line)
            elif line[1] =='1':
                val_img_list.append(line)
            else:
                test_img_list.append(line)

        f.close()

        # state is passed as optional argument (default='train')
        if state=='train':
            train_img_list.sort()
            self.image_list = train_img_list
        elif state=='val':
            val_img_list.sort()
            self.image_list = val_img_list
        else:
            test_img_list.sort()
            self.image_list = test_img_list


        ## DEFINE PRE PROCESSING STRATEGY: DATA AUGMENTATION OR NOT
        # pre-processing is only applied to 128x128 target image
        if state=='train' and self.data_augmentation:
            self.pre_process = transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop((178, 178)),
                                                transforms.Resize((128, 128)),
                                                transforms.RandomRotation(20, resample=Image.BILINEAR),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                                               ])
        else:
            self.pre_process = transforms.Compose([
                                            transforms.CenterCrop((178, 178)),
                                            transforms.Resize((128,128)),
                                            ])

        ## DEFINE totensor OPERATION. FIRST MAKE IT TENSOR (as usual), THEN NORMALIZE
        self.totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        ## DEFINE THREE SCALING FOR PROGRESSIVE TRAINING
        self._64x64_down_sampling = transforms.Resize((64, 64))
        self._32x32_down_sampling = transforms.Resize((32, 32))
        self._16x16_down_sampling = transforms.Resize((16,16))

    # https://stackoverflow.com/questions/43627405/understanding-getitem-method
    # this overwrites the built-in __getitem__ method
    # for this to take effect: dataloader = CelebDataSet() \ dataloader[i]
    def __getitem__(self, index):
        image_path = join(self.img_path, self.image_list[index][0])
        
        # RGB 128x128 original image
        target_image = Image.open(image_path).convert('RGB')
        target_image = self.pre_process(target_image)
        target_image = self.totensor(target_image)                  # normalization

        # input image is the orginal, downsampled at 16x16
        input_image = self._16x16_down_sampling(x2_target_image)
        input_image = self.totensor(input_image)                    # normalization
        
        # original downsampled at 32x32. This'll be compared with input_image 2x upsampling
        x2_target_image = self._32x32_down_sampling(x4_target_image)
        x2_target_image = self.totensor(x2_target_image)

        # original downsampled at 64x64. This'll be compared with x2_target_image 2x upsampling
        x4_target_image = self._64x64_down_sampling(target_image)
        x4_target_image = self.totensor(x4_target_image)            # normalization

        return x2_target_image, x4_target_image, target_image, input_image

    def __len__(self):
        return len(self.image_list)
