import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, psnr, SSIMLoss, ssim, convert_rgb_to_y, convert_rgb_to_ycbcr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default="HanokModel/srcnnx2_rgb.h5")
    parser.add_argument('--eval-file', type=str, default="HanokModel/srcnnEvalx2_rgb.h5")
    parser.add_argument('--outputs-dir', type=str, default="HanokModel/model/srcnn")
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', type=str, default="MSE")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--fine-tuning', type=bool, default=False)
    parser.add_argument('--pretrained-model', type=str, default="srcnn_x2.pth")
    parser.add_argument('--channel-number', type=int, default="3")
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN(num_channels=args.channel_number, tuning=args.fine_tuning).to(device)
    if args.fine_tuning:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))

    criterion = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights_psnr = copy.deepcopy(model.state_dict())
    best_weights_ssim = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    best_ssim = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                if args.channel_number == 1:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                else:
                    inputs = inputs.squeeze(1)
                    labels = labels.squeeze(1)

                pred = model(inputs)

                if args.loss == "MSE":
                    loss = criterion(pred, labels)
                elif args.loss == "SSIM":
                    loss = SSIMLoss(pred, labels, device, args.channel_number)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        if args.channel_number == 1:
            for data in eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    pred = model(inputs).clamp(0.0, 1.0)

                epoch_psnr.update(psnr(pred, labels, args.channel_number), len(inputs))
                epoch_ssim.update(ssim(pred, labels, args.channel_number), len(inputs))
        else:
            for data in eval_dataloader:
                inputs, labels = data

                input = inputs[0][0].transpose(2, 0).transpose(2, 1).unsqueeze(0).to(device)
                labels = labels[0][0].cpu().numpy()
                with torch.no_grad():
                    pred = model(input).squeeze().cpu().numpy().transpose(1, 2, 0)

                epoch_psnr.update(psnr(pred, labels, args.channel_number), len(inputs))
                epoch_ssim.update(ssim(pred, labels, args.channel_number), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        print('eval ssim: {:.2f}'.format(epoch_ssim.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights_psnr = copy.deepcopy(model.state_dict())

        if epoch_ssim.avg > best_ssim:
            best_epoch = epoch
            best_ssim = epoch_ssim.avg
            best_weights_ssim = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights_psnr, os.path.join(args.outputs_dir, 'best_psnr.pth'))

    print('best epoch: {}, ssim: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights_ssim, os.path.join(args.outputs_dir, 'best_ssim.pth'))
