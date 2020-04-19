""" Some explanations

# https://stackoverflow.com/questions/52798540/working-with-ssim-loss-function-in-tensorflow-for-rgb-images
    SSIM should measure the similarity between my reconstructed output image of my denoising autoencoder 
    and the input uncorrupted image (RGB).
"""

import torch
from torch import optim, nn
import argparse
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable, grad
import sys
from torchvision import utils
from math import log10
from ssim import ssim, msssim

# these are my two scripts
from model import Generator
from dataloader import CelebDataSet

def test(dataloader, generator, MSE_Loss, step, alpha):
    avg_psnr = 0
    avg_ssim = 0
    avg_msssim = 0
    
    # https://stackoverflow.com/a/24658101
    for i, (x2_target_image, x4_target_image, target_image, input_image) in enumerate(dataloader):
        
        input_image = input_image.to(device)
        
        if step==1:
            target_image = x2_target_image.to(device)
        elif step==2:
            target_image = x4_target_image.to(device)
        else:
            target_image = target_image.to(device)

        # define input image
        input_image = input_image.to(device)
        
        # make a prediction
        predicted_image = generator(input_image, step, alpha)
        predicted_image = predicted_image.double()
        
        # retrieve the original image and compute losses
        target_image = target_image.double()
        mse_loss = MSE_Loss(0.5*predicted_image+0.5, 0.5*target_image+0.5)
        psnr = 10*log10(1./mse_loss.item())
        avg_psnr += psnr
        _ssim = ssim(0.5*predicted_image+0.5, 0.5*target_image+0.5)
        avg_ssim += _ssim.item()
        ms_ssim = msssim(0.5*predicted_image+0.5, 0.5*target_image+0.5)
        avg_msssim += ms_ssim.item()

        sys.stdout.write('\r [%d/%d] Test progress... PSNR: %6.4f'%(i, len(dataloader), psnr))
        save_image = torch.cat([predicted_image, target_image], dim=0)
        if args.local_rank==0:
            utils.save_image(0.5*save_image+0.5, os.path.join(args.result_path, '%d_results.jpg'%i))
    print('Test done, Average PSNR:%6.4f, Average SSIM:%6.4f, Average MS-SSIM:%6.4f '%(avg_psnr/len(dataloader),avg_ssim/len(dataloader), avg_msssim/len(dataloader)))


if __name__ == '__main__':
    ## PARSER
    parser = argparse.ArgumentParser('Implemnetation of Progressive Face Super-Resolution Attention to Face Landmarks')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--checkpoint-path', default='./checkpoints/', type=str)
    parser.add_argument('--data-path', default='./dataset/', type=str)
    parser.add_argument('--result-path', default='./result/', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()

    ## MAKE DIR IF NECESSARY
    if args.local_rank == 0:
        if not os.path.exists(args.result_path):
            os.mkdir(args.result_path)
            print('===>make directory', args.result_path)

    ## SELECT DEVICE
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args.gpu = 0
    args.world_size = 1

    # here the dataset is thought only for testing! we are NOT training
    # we are preparing the test dataset
    dataset = CelebDataSet(data_path=args.data_path, state='test')
    
    # this is run ONLY if "distributed" command is specified
    if args.distributed:
        import apex.parallel as parallel
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    else:
        # load the dataset using structure DataLoader (part of torch.utils.data)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # instantiate Generator(nn.Module) and load in cpu/gpu
    generator = Generator().to(device)
    
    ## DEFINE CHECKPOINT
    # checkpoints are used during training to save a model (model parameters I suppose)
    # here we are only testing the pre-trained model, thus we load (torch.load) the model
    if args.distributed:
        g_checkpoint = torch.load(args.checkpoint_path, map_location = lambda storage, loc: storage.cuda(args.local_rank))
        generator = parallel.DistributedDataParallel(generator)
        generator = parallel.convert_syncbn_model(generator)
    else:
        g_checkpoint = torch.load(args.checkpoint_path)
    
    generator.load_state_dict(g_checkpoint['model_state_dict'], strict=False)
    step = g_checkpoint['step']
    alpha = g_checkpoint['alpha']
    iteration = g_checkpoint['iteration']
    print('pre-trained model is loaded step:%d, alpha:%d iteration:%d'%(step, alpha, iteration))
    MSE_Loss = nn.MSELoss()

    # notify all layers that you are in eval mode instead of training mode
    generator.eval()

    test(dataloader, generator, MSE_Loss, step, alpha)
