import torch
import argparse
import os
from model import Generator
from PIL import Image
import torchvision.transforms as transforms
from torchvision import utils

from dataloader import CelebDataSet
from compare import Comparison
from transformations import Transformations

if __name__ == '__main__':
    ## PARSER
    parser = argparse.ArgumentParser('Demo of Progressive Face Super-Resolution')
    parser.add_argument('--image-id', type=int, default=None)
    parser.add_argument('--image-dirpath', type=str, default=None)
    parser.add_argument('--checkpoint-path', default='./checkpoints/generator_checkpoint_singleGPU.ckpt')
    args = parser.parse_args()

    output_path = './OUTPUT_PATH'

    # check if using GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('USING CUDA')
    else:
        device = torch.device('cpu')
        print('found only gpu')
    
    # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations
    with torch.no_grad():
        # the generator is a sub-class of torch.nn.Module, basic class for NNs
        # Module can contain sub-modules, in this case Generator (defined in model)
        # to is used to call
        generator = Generator().to(device)
        generator.eval()    # notify all layers that you are in eval mode instead of training mode
        g_checkpoint = torch.load(args.checkpoint_path)
        generator.load_state_dict(g_checkpoint['model_state_dict'], strict=False)
        step = g_checkpoint['step']
        alpha = g_checkpoint['alpha']
        iteration = g_checkpoint['iteration']
        print('pre-trained model is loaded step:%d, alpha:%d iteration:%d'%(step, alpha, iteration))

        # if you want to work with CelebA dataset
        if args.image_id is not None:
            dataset = CelebDataSet(data_path='./dataset')       # init dataset
            img_to_SR = dataset.getPersonPath(args.image_id)    # choose the image
            DF_people = dataset.getPeopleDF()                   # people DF
        
        # if you want to work with a random image uploaded
        if args.image_dirpath is not None:
            img_to_SR = args.image_dirpath

        transfo = Transformations()                         # init transform.
        x2_target_image, x4_target_image, target_image, input_image = transfo.perform(img_to_SR)

        # load 16x16 image to device
        input_image = input_image.unsqueeze(0).to(device)

        # crete output dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # generate and save image
        output_filename = os.path.join(output_path, 'imageID_{}.jpg'.format(str(args.image_id)))
        output_image = generator(input_image, step, alpha)
        utils.save_image(0.5*output_image+0.5, output_filename)

        if args.image_id is not None:
            comp = Comparison(output_filename, DF_people, args.image_id)
            comp.compare()
        
            printable = comp.getResult()
            for k, v in printable.items():
                print(k, v)