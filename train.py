import torch
import argparse
import os
import sys
import apex.parallel as parallel
from torch import optim, nn
from torchvision import utils
from torch.autograd import Variable, grad
from dataloader import CelebDataSet
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from utils import inf_train_gen, requires_grad, get_heat_map, operate_heatmap
from vggnet import Vgg16
from face_alignment.models import FAN
from face_alignment.utils import *

def train(generator, discriminator, face_align_net, g_optim, d_optim, input_image, target_image, step, iteration, alpha):
    
    #generator training
    requires_grad(generator, True)
    requires_grad(discriminator, False)
    
    generator.zero_grad()
    fake_image = generator(input_image, step, alpha)
    pixel_loss = L1_Loss(fake_image, target_image)
    predict_fake = discriminator(fake_image, step, alpha)

    feature_real = vgg(0.5*target_image+0.5)
    feature_fake = vgg(0.5*fake_image+0.5)

    perceptual_loss=0
    for fr, ff in zip(feature_real, feature_fake):
        perceptual_loss += MSE_Loss(ff, fr) #MSE_Loss_sum(ff, fr)

    if step==1:
        g_loss = - predict_fake.mean()+ 10*pixel_loss + 1e-2*perceptual_loss
        
    elif step>1:
        hm_f, hm_r = get_heat_map(face_align_net, fake_image, target_image, False, 2**(4-step))
        face_hmap_loss = MSE_Loss(hm_f, hm_r)
        heatMap = operate_heatmap(hm_r)
        diff = abs(fake_image - target_image)
        attention_loss = torch.mean(heatMap*diff)
        if step==2:
            g_loss = 10*pixel_loss - predict_fake.mean() + 1e-2*perceptual_loss + 500*attention_loss + 50*face_hmap_loss
        else:
            g_loss = 10*pixel_loss - 1e-1*predict_fake.mean() + 1e-3*perceptual_loss + 50*attention_loss + 50*face_hmap_loss
    
    g_loss.backward()
    g_optim.step()

    #discriminator training
    requires_grad(generator, False)
    requires_grad(discriminator, True)
    discriminator.zero_grad()
    predict_real = discriminator(target_image, step, alpha)
    predict_real = predict_real.mean() - 0.001 * (predict_real ** 2).mean()
    fake_image = generator(input_image, step, alpha)
    predict_fake = discriminator(fake_image, step, alpha)
    predict_fake = predict_fake.mean()

    #Gradient Penalty (GP)
    eps = torch.rand(input_image.size(0), 1, 1, 1).to(device)
    x_hat = eps * target_image.data + (1-eps) * fake_image.data
    x_hat = Variable(x_hat, requires_grad=True).to(device)
    hat_predict = discriminator(x_hat, step, alpha)
    grad_x_hat = grad(
        outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
    grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) -1)**2).mean()
    #grad_penalty = torch.max(torch.zeros(1).to(device)
    #        , ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) -1)).mean())
    grad_penalty = 10 * grad_penalty
    
    d_loss = predict_fake - predict_real + grad_penalty

    d_loss.backward()
    d_optim.step()

    if args.local_rank==0:
        if step==1:
            sys.stdout.write('\r Step:%1d Iteration:%5d alpha:%6.5f Pixel_loss:%6.4f perceptual_loss:%6.4f Generator loss:%6.4f Discriminator loss:%6.4f'\
            %(step, iteration, alpha, pixel_loss.item(), perceptual_loss.item(), g_loss.item(), d_loss.item()))
        else:
            sys.stdout.write('\r Step:%1d Iteration:%5d alpha:%6.5f Pixel_loss:%6.4f perceptual_loss:%6.4f attention_loss:%6.4f face_hmap_loss:%6.4f Generator loss:%6.4f Discriminator loss:%6.4f'\
            %(step, iteration, alpha, pixel_loss.item(), perceptual_loss.item(), attention_loss.item(), face_hmap_loss.item(), g_loss.item(), d_loss.item()))

        #save predict sample
        if iteration % 100 == 0:
            imgs = torch.cat([0.5*fake_image+0.5, 0.5*target_image+0.5], dim=0)
            utils.save_image(imgs, os.path.join(args.result_path,'result_iteration{}.jpeg'.format(iteration)))

        if iteration%args.save_interval==0:
            torch.save({
                        'step':step,
                        'alpha':alpha,
                        'iteration':iteration,
                        'model_state_dict': generator.state_dict(),
                        }, os.path.join(args.checkpoint_path,'awgan_generator_checkpoint_{}.ckpt'.format((step-1)*50000+iteration)))
            torch.save({
                        'step':step,
                        'alpha':alpha,
                        'iteration':iteration,
                        'model_state_dict': discriminator.state_dict(),
                        }, os.path.join(args.checkpoint_path+'awgan_discriminator_checkpoint_{}.ckpt'.format((step-1)*50000+iteration)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Implemnetation of Progressive Face Super-Resolution Attention to Face Landmarks')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--pre-train', action='store_true')
    parser.add_argument('--checkpoint-path', default='./checkpoints/', type=str)
    parser.add_argument('--data-path', default='./dataset/', type=str)
    parser.add_argument('--result-path', default='./result/', type=str)
    parser.add_argument('--max-iter', default=200000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--save_interval', default=5000, type=int)
    parser.add_argument('--last-iter', type=int, default=100000)
    parser.add_argument('--gpu-ids', default=[0,1,2,3], nargs='+', type=int)
    parser.add_argument('--step-iteration', default=50000, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--data-augment', default=False, action='store_true')
    args = parser.parse_args()

    if args.local_rank==0:
        print(args)
        if not os.path.exists(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)
            print('===>make directory', args.checkpoint_path)
        if not os.path.exists(args.result_path):
            os.mkdir(args.result_path)
            print('===>make directory', args.result_path)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.gpu = 0
    args.world_size = 1

    dataset = CelebDataSet(data_path=args.data_path, state='train', data_augmentation=args.data_augment)

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    dataloader = inf_train_gen(dataloader)

    ### load models in memory
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg = Vgg16(requires_grad=False).to(device)

    ### if want to use a pretrained model
    if args.pre_train:
        print('loading pretrained, fix this')
        if args.distributed:
            g_checkpoint = torch.load(args.checkpoint_path+'generator_checkpoint_{}.ckpt'.format(args.last_iter), map_location = lambda storage, loc: storage.cuda(args.local_rank))
            d_checkpoint = torch.load(args.checkpoint_path+'discriminator_checkpoint_{}.ckpt'.format(args.last_iter), map_location = lambda storage, loc: storage.cuda(args.local_rank))
        else:
            g_checkpoint = torch.load(args.checkpoint_path+'generator_checkpoint_{}.ckpt'.format(args.last_iter))
            d_checkpoint = torch.load(args.checkpoint_path+'discriminator_checkpoint_{}.ckpt'.format(args.last_iter))
        
        generator.load_state_dict(g_checkpoint['model_state_dict'], strict=False)
        discriminator.load_state_dict(d_checkpoint['model_state_dict'], strict=False)
        step = g_checkpoint['step']
        alpha = g_checkpoint['alpha']
        iteration = g_checkpoint['iteration']
        print('pre-trained model is loaded step:%d, iteration:%d'%(step, iteration))
    else:
        iteration = 0
        step = 1

    if args.distributed:
        generator = parallel.DistributedDataParallel(generator)
        discriminator = parallel.DistributedDataParallel(discriminator)
        vgg = parallel.DistributedDataParallel(vgg)
        face_align_net = parallel.DistributedDataParallel(
            torch.load('./checkpoints/compressed_model_011000.pth', map_location = lambda storage, loc: storage.cuda(args.local_rank)).to(device)
            )
    else:
        if len(args.gpu_ids) >1:
            generator = nn.DataParallel(generator, args.gpu_ids)
            discriminator = nn.DataParallel(discriminator, args.gpu_ids)
            vgg = nn.DataParallel(vgg, args.gpu_ids)
            face_align_net = nn.DataParallel(torch.load('./checkpoints/compressed_model_011000.pth').to(device), args.gpu_ids)
        else:
            face_align_net =torch.load('./checkpoints/compressed_model_011000.pth').to(device)


    ### FINE TUNING (START)
    generator.load_state_dict(g_checkpoint['model_state_dict'], strict=False)
    step = g_checkpoint['step']
    alpha = g_checkpoint['alpha']
    iteration = g_checkpoint['iteration']

    # fine tuning generator
    for param in generator.parameters():
        param.requires_grad = False

    generator.step3 = nn.Sequential([nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
                                     nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
                                     nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)])

    ### FINE TUNING (END)

    g_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.,0.99)) #beta WGAN refered
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0., 0.99))

    MSE_Loss = nn.MSELoss().to(device)
    MSE_Loss_sum = nn.MSELoss(reduction='sum').to(device)
    L1_Loss = nn.L1Loss().to(device)

    alpha_weight = float(2./args.step_iteration)

    for i in range(args.max_iter-iteration):
        if iteration>=args.step_iteration:
            if step < 3:
                alpha = 0
                iteration = 0
                step += 1
        alpha = min(1, alpha_weight * iteration)
        try:
            dat = dataloader.__next__()
            x2_target_image, x4_target_image, x8_target_image, input_image = dat
        except (OSError, StopIteration):
            dat = dataloader.__next__()
            x2_target_image, x4_target_image, x8_target_image, input_image = dat
        iteration +=1
        input_image = input_image.to(device)
        if step==1:
            target_image = x2_target_image.to(device)
        elif step==2:
            target_image = x4_target_image.to(device)
        elif step==3:
            target_image = x8_target_image.to(device)

        train(generator, discriminator, face_align_net, g_optim, d_optim, input_image, target_image, step, iteration, alpha)