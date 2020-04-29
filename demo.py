import torch
import argparse
from model import Generator
from PIL import Image
import torchvision.transforms as transforms
from torchvision import utils

if __name__ == '__main__':
    ## PARSER
    parser = argparse.ArgumentParser('Demo of Progressive Face Super-Resolution')
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--checkpoint-path', default='./checkpoints/generator_checkpoint_singleGPU.ckpt')
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()

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

        input_image = Image.open(args.image_path).convert('RGB')

        _16x16_down_sampling = transforms.Resize((16,16))
        _64x64_down_sampling = transforms.Resize((64, 64))
        _32x32_down_sampling = transforms.Resize((32, 32))

        totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        #Note: Our netowork is trained by progressively downsampled images.
        transformed_image = _16x16_down_sampling(_32x32_down_sampling(_64x64_down_sampling(input_image)))
        transformed_image = totensor(transformed_image).unsqueeze(0).to(device)

        output_image = generator(transformed_image, step, alpha)

        utils.save_image(0.5*output_image+0.5, args.output_path)

