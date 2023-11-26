import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_ycbcr_to_rgb, convert_rgb_to_y, psnr, convert_rgb_to_ycbcr, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default="best_psnr.pth")
    parser.add_argument('--image-file', type=str, default="hanok35.jpg")
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--channel', type=str, default="RGB")
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.channel == "Y":
        model = SRCNN(num_channels=1).to(device)
    elif args.channel == "RGB":
        model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    image = np.array(image).astype(np.float32)
    if args.channel == "Y":
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        psnr = calc_psnr(y, preds)
        print('PSNR: {:.2f}'.format(psnr))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    elif args.channel == "RGB":
        y1 = convert_rgb_to_y(image)
        y1 = torch.from_numpy(y1).to(device)

        image = image.transpose((2, 0, 1))
        with torch.no_grad():
            preds = model(torch.from_numpy(image).unsqueeze(0).to(device))
        output = preds.squeeze().cpu().numpy().transpose(1, 2, 0)
        y2 = convert_rgb_to_y(preds)
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        print('PSNR: {:.2f}'.format(psnr(y1, y2, 1)))

    output = pil_image.fromarray(output)

    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))
