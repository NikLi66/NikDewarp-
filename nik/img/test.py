import argparse
import logging
import os

import numpy as np
import torch
import pickle as pk

from nik.img.unet import ImgUNet
from nik.utils.util_img_io import save_img, dewarp
from nik.utils.util_logger import logging_to_file, string_monitor
from nik.utils.util_model import load_checkpoint
from nik.utils.util_img_io import get_img_offset

import cv2


def read_file(img_path, id):
    with open(img_path, 'rb') as f:
        perturbed_data = pk.load(f)

    img = perturbed_data[id]['img']  # HWC [1024, 960, 3]
    fp = perturbed_data[id]['fp']  # WHC [693, 1012, 2]
    ori = perturbed_data[id]['ori']  # HWC [1012, 693, 3]

    return img, fp, ori


def read_img(img_path):
    src = cv2.imread(img_path, 1)
    return src


class DataProcess():
    def __init__(self, img_height, img_width):
        self._img_height = img_height
        self._img_width = img_width
        self._img_offset = get_img_offset(self._img_height, self._img_width)

    def process(self, img, fp):
        # change fp to opecv format  HWC
        fp = fp.transpose(1, 0, 2).astype(np.float32)

        # fp
        fp = cv2.resize(fp, (self._img_width, self._img_height), interpolation=cv2.INTER_LINEAR)
        # img
        scale3 = self._img_height / img.shape[0]
        scale4 = self._img_width / img.shape[1]
        img = cv2.resize(img, (self._img_width, self._img_height), interpolation=cv2.INTER_LINEAR)
        fp[:, :, 0] = fp[:, :, 0] * scale4  # width
        fp[:, :, 1] = fp[:, :, 1] * scale3  # height
        fp = fp - self._img_offset  # change to offset

        # change to gray
        img = np.mean(img, axis=2, keepdims=True)

        return img, fp

def transform_canny(img, img_height, img_width):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_magnitude = np.reshape(gradient_magnitude, [img_height, img_width, 1]).transpose(2, 0, 1).astype(
        np.float32)

    return gradient_magnitude


def run_demo(args):
    """
    img_path: 矫正图片地址
    output_dir: 输出地址
    use_gpu: 默认为1
    model_name: 默认为unet3，全部见nik.img.unet.py
    checkpoint: 训练好的模型地址
    img_height: 默认448，与训练一致
    img_height: 默认448，与训练一致
    """
    dataprocess = DataProcess(img_height=args.img_height, img_width=args.img_width)
    img = read_img(args.img_path)
    fp = np.zeros([img.shape[1], img.shape[0], 2])
    img, fp = dataprocess.process(img=img, fp=fp)

    # backup img and ori
    np_img = img

    # 检测边缘
    fp = cv2.resize(fp, [args.img_height // 4, args.img_width // 4])

    img = img.transpose(2, 0, 1).astype(np.float32)
    fp = fp.transpose(2, 0, 1).astype(np.float32)

    img = torch.from_numpy(img)
    fp = torch.from_numpy(fp)

    # log
    logging_to_file(os.path.join(args.output_dir, "log.txt"))
    logger = logging.getLogger()

    # gpu / cpu
    if args.use_gpu > 0 and torch.cuda.is_available():
        logger.info('train with gpu {} and pytorch {}'.format(args.use_gpu, torch.__version__))
        device = torch.device("cuda")
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    in_channels = 1
    if args.use_canny_input == 2:
        in_channels = in_channels * 2
    if args.use_ori == 1:
        in_channels = in_channels * 2
    model = ImgUNet(in_channels=in_channels, out_channels=2, use_ori=0,
                       use_canny_input=0, model_name=args.model_name)

    model = model.to(device)
    model.eval()
    start_epoch = load_checkpoint(args.checkpoint, model)
    logger.info('load checkpoint from {}'.format(args.checkpoint))

    img, fp = img.unsqueeze(0), fp.unsqueeze(0)

    output = model(img.to(device))

    # to numpy
    np_fp = output.cpu().detach().numpy()[0].transpose(1, 2, 0)
    np_fp = cv2.resize(np_fp, (args.img_height, args.img_width), interpolation=cv2.INTER_LINEAR)
    if args.use_offset == 1:
        np_fp = np_fp + dataprocess._img_offset

    np_fp = np.clip(np_fp, 0, args.img_height - 1)
    new_img = dewarp(img=np_img, fp=np_fp)

    print(new_img.shape)

    save_img(img_path=os.path.join(args.output_dir, "img_user.jpg"), array=np_img)
    save_img(img_path=os.path.join(args.output_dir, "dewarp_%s.jpg" % args.model_name), array=new_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str,
                        default='test.jpg',
                        help='img path')
    parser.add_argument('--output_dir', type=str,
                        default='./test_output/',
                        help='output dir')
    parser.add_argument('--checkpoint', type=str,
                        default='',
                        help='init')
    parser.add_argument('--use_gpu', type=int,
                        default=1,
                        help='multi gpu')
    parser.add_argument('--img_height', type=int,
                        default=448,
                        help='img height')
    parser.add_argument('--img_width', type=int,
                        default=448,
                        help='img width')
    parser.add_argument('--model_name', type=str,
                        default='unet3',
                        help='model name')
    parser.add_argument('--use_offset', type=int,
                        default=1,
                        help='using offset')

    args, _ = parser.parse_known_args()

    run_demo(args)
