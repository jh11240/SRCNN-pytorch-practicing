import torch
import numpy as np


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def rgb_psnr(img1, img2):
    img1 = convert_rgb_to_y(img1)
    img2 = convert_rgb_to_y(img2)
    if str(type(img1)) != "<class 'numpy.ndarray'>":
        img1 = img1.detach().numpy()
    if str(type(img2)) != "<class 'numpy.ndarray'>":
        img2 = img2.detach().numpy()
    return 10. * np.log10(255. / np.mean((img1 - img2) ** 2)) / 3

def psnr(img1, img2, channel_num):
    if channel_num == 3:
        return rgb_psnr(img1, img2)
    elif channel_num == 1:
        return calc_psnr(img1, img2)

    return 0


def rgb_ssim(img1, img2):
    ssim = 0
    if str(type(img1)) != "<class 'numpy.ndarray'>":
        img1 = img1.detach().numpy()
    if str(type(img2)) != "<class 'numpy.ndarray'>":
        img2 = img2.numpy()

    for i in range(3):
        avg_x = np.mean(img1[:, :, i])
        avg_y = np.mean(img2[:, :, i])

        std_x = np.std(img1[:, :, i])
        std_y = np.std(img2[:, :, i])

        cov_xy = np.cov(img1[:, :, i].flatten(), img2[:, :, i].flatten())[0, 1]

        ssim += (2 * avg_x * avg_y + 6.5025) * (2 * cov_xy + 58.5225) / (avg_x ** 2 + avg_y ** 2 + 6.5025) / (
                    std_x ** 2 + std_y ** 2 + 58.5225)

    return ssim / 3


def grayscale_ssim(img1, img2):
    img1 = img1.detach().numpy()
    img2 = img2.numpy()

    img1 = (img1 * 255).astype(np.uint8)

    img2 = (img2 * 255).astype(np.uint8)

    avg_x = np.mean(img1)
    avg_y = np.mean(img2)

    std_x = np.std(img1)
    std_y = np.std(img2)

    cov_xy = np.cov(img1.flatten(), img2.flatten())[0, 1]

    ssim = (2 * avg_x * avg_y + 6.5025) * (2 * cov_xy + 58.5225) / (avg_x ** 2 + avg_y ** 2 + 6.5025) / (
                std_x ** 2 + std_y ** 2 + 58.5225)

    return ssim


def ssim(img1, img2, channel_num):
    if channel_num == 3:
        return rgb_ssim(img1, img2)
    elif channel_num == 1:
        return grayscale_ssim(img1, img2)

    return 0


def SSIMLoss(Y_pred, Y, device, channel_num):
    loss = 1 - ssim(Y_pred, Y, channel_num)

    loss_tensor = torch.tensor(loss, requires_grad=True).to(device)

    return loss_tensor


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
