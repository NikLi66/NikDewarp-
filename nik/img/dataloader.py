import logging
import os.path
import pickle
import random
import time
from threading import Lock, Thread

import cv2
import numpy as np
import torch
import torch.utils.data as Data

from nik.utils.util_img_io import get_img_offset
from nik.utils.util_zip import zip_decode


def crop_image_tight(img, grid2D, rand_edge):
    """
    Crops the image tightly around the keypoints in grid2D.
    This function creates a tight crop around the document in the image.
    """
    size = img.shape

    minx = np.floor(np.amin(grid2D[:, :, 0])).astype(int)
    maxx = np.ceil(np.amax(grid2D[:, :, 0])).astype(int)
    miny = np.floor(np.amin(grid2D[ :, :, 1])).astype(int)
    maxy = np.ceil(np.amax(grid2D[:, :, 1])).astype(int)
    if minx < 0 or miny < 0 or maxx > size[1] or maxy > size[0]:
        minx = max(minx, 0)
        miny = max(miny, 0)
        maxx = min(maxx, size[1])
        maxy = min(maxy, size[0])

    if rand_edge == 1:
        s = 20
        s = min(min(s, minx), miny)  # s shouldn't be smaller than actually available natural padding is
        s = min(min(s, size[1] - 1 - maxx), size[0] - 1 - maxy)

        # Crop the image slightly larger than necessary
        img = img[miny - s : maxy + s, minx - s : maxx + s]
        cx1 = random.randint(0, max(s - 5, 1))
        cx2 = random.randint(0, max(s - 5, 1)) + 1
        cy1 = random.randint(0, max(s - 5, 1))
        cy2 = random.randint(0, max(s - 5, 1)) + 1

        img = img[cy1:-cy2, cx1:-cx2]
        top = miny - s + cy1
        bot = size[0] - maxy - s + cy2
        left = minx - s + cx1
        right = size[1] - maxx - s + cx2
    else:
        img = img[miny: maxy, minx: maxx]
        top = miny
        bot = size[0] - maxy
        left = minx
        right = size[1] - maxx

    # print(top, bot, left, right )
    return img, top, bot, left, right



def crop_tight(img_RGB, grid2D, size, rand_edge=0):
    # The incoming grid2D array is expressed in pixel coordinates (resolution of img_RGB before crop/resize)
    size_raw = img_RGB.shape
    img, top, bot, left, right = crop_image_tight(img_RGB, grid2D, rand_edge)
    # print(img.shape)
    img = cv2.resize(img, size)

    grid2D[:, :, 0] = (grid2D[:, :, 0] - left) * size[1] / (size_raw[1] - left - right)
    grid2D[:, :, 1] = (grid2D[:, :, 1] - top) * size[0] / (size_raw[0] - top - bot)

    return img, grid2D

def crop_image_border(img):
    height, width = img.shape
    ww = np.mean(img, axis=0)
    shield = 2
    left = 0
    for i in range(width):
        if ww[left] < shield:
            left = i
        else:
            break
    right = 0
    for i in range(width):
        if ww[width - 1 - i] < shield:
            right = i
        else:
            break
    hh = np.mean(img, axis=1)
    top = 0
    for i in range(height):
        if hh[top] < shield:
            top = i
        else:
            break
    bot = 0
    for i in range(height):
        if hh[height - 1 - i] < shield:
            bot = i
        else:
            break
    # print(top, bot, left, right )
    return img[top:height-bot, left:width-right], top, bot, left, right

def crop_border(img_RGB, grid2D, size):
    size_raw = img_RGB.shape
    img, top, bot, left, right = crop_image_border(img_RGB)
    # print(img.shape)
    img = cv2.resize(img, size)

    grid2D[:, :, 0] = (grid2D[:, :, 0] - left) * size[1] / (size_raw[1] - left - right)
    grid2D[:, :, 1] = (grid2D[:, :, 1] - top) * size[0] / (size_raw[0] - top - bot)

    return img, grid2D

class NikDataSet(Data.Dataset):
    def __init__(self, img_height, img_width, use_ori, use_canny_input, data_list_path, is_shuffle=True, num_workers=1):
        self._is_shuffle = is_shuffle
        self._logger = logging.getLogger()

        self._img_height = img_height
        self._img_width = img_width
        self._img_offset = get_img_offset(img_height=img_height, img_width=img_width)
        self._img_offset_large = get_img_offset(img_height=1024, img_width=1024)

        self._use_ori = use_ori
        self._use_canny_input = use_canny_input
        self._fp_scale = 4

        self._fp_height = img_height // self._fp_scale
        self._fp_width = img_width // self._fp_scale

        self._img_height_raw = 1024
        self._img_width_raw = 1024
        self._fp_height_raw = 256
        self._fp_width_raw = 256
        self._img_offset_raw = get_img_offset(img_height=self._img_height_raw, img_width=self._img_width_raw)

        with open(data_list_path, 'r') as f:
            pk_path_list = list(line.strip() for line in f if line)

        self._pk_list_in = []

        for i, line in enumerate(pk_path_list):
            with open(line, "rb") as f:
                self._pk_list_in.append(pickle.load(f))

        if self._is_shuffle:
            random.shuffle(self._pk_list_in)


    def __len__(self):
        return len(self._pk_list_in)

    def __getitem__(self, id):
        data = self._pk_list_in[id]
        img, fp = data["image"], data["fiducial_points"]
        return self.process(img, fp)

    def process(self, img, fp):
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_ANYCOLOR)
        if len(img.shape) > 2 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fp = zip_decode(fp, np.int16, [self._fp_height_raw, self._fp_width_raw, 2]).astype(np.float32)

        #trans to raw
        fp = cv2.resize(fp, [self._img_width_raw, self._img_height_raw])
        fp = fp + self._img_offset_raw
        img, fp = crop_tight(img, fp, [self._img_width_raw, self._img_height_raw], rand_edge=0)

        # random crop
        start_x = random.randint(0, self._img_width_raw - self._img_width)
        start_y = random.randint(0, self._img_height_raw - self._img_height)
        end_x = random.randint(start_x + self._img_width, self._img_width_raw)
        end_y = random.randint(start_y + self._img_height, self._img_height_raw)
        img = img[start_y:end_y, start_x:end_x]
        mask_x = np.logical_and(fp[:, :, 0] > start_x, fp[:, :, 0] < end_x)
        mask_y = np.logical_and(fp[:, :, 1] > start_y, fp[:, :, 1] < end_y)
        mask_xy = np.logical_and(mask_x, mask_y)
        points = np.argwhere(mask_xy)
        fp -= [start_x, start_y]

        # key step: 求外接矩形
        x,y,w,h = cv2.boundingRect(points)
        fp = fp[x:x+w, y:y+h, :]
        # 此时fp中有负数
        # mask_xy = mask_xy[x:x+w, y:y+h].astype(np.float32)
        # print(x, y, w, h, start_x, start_y)

        fp = cv2.resize(fp, [self._img_width, self._img_height])
        # mask_xy = cv2.resize(mask_xy, [self._img_width, self._img_height])
        fp = (fp/[end_x-start_x, end_y-start_y] * [self._img_width, self._img_height]).astype(np.float32)
        img = cv2.resize(img, [self._img_width, self._img_height])

        # calc mask
        warp = cv2.remap(img, fp[:, :, 0], fp[:, :, 1], interpolation=cv2.INTER_LINEAR)
        warp_tmp = cv2.resize(warp, [self._img_width, self._img_height])
        mask = cv2.Canny(warp_tmp, threshold1=50, threshold2=150)
        # 定义膨胀操作的核大小
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1) / 255
        # mask = mask + 1 - mask_xy
        mask = mask[:mask.shape[0]:self._fp_scale, :mask.shape[1]:self._fp_scale]



        #tran to new
        if random.randint(1, 10) < 3:
            #[1,2]
            img = warp
            fp = np.zeros([self._fp_height, self._fp_width, 2])
        else:
            fp = fp - self._img_offset
            fp = cv2.resize(fp, [self._fp_width, self._fp_height])


        #to torch
        mask = np.reshape(mask, [1, self._fp_height, self._fp_width]).astype(np.float32)
        fp = np.reshape(fp, [self._fp_height, self._fp_width, 2]).transpose(2, 0, 1).astype(np.float32)
        img = np.reshape(img, [1, self._img_height, self._img_width]).astype(np.float32)


        fp = torch.from_numpy(fp)
        mask = torch.from_numpy(mask)
        img = torch.from_numpy(img)


        return img, fp, mask

    def transform_canny(self,img):

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradient_magnitude = np.reshape(gradient_magnitude, [1, self._img_height, self._img_width]).astype(np.float32)

        return gradient_magnitude

    def transform_horizontal(self, img, scale=0.3):
        h, w = img.shape[:2]
        src, dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"), \
            np.array([[0, random.randint(0, int(scale*h))], [w-1, 0],
                      [w - 1, h - 1], [0, random.randint(int((1-scale)*h), h - 1)]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        res = cv2.warpPerspective(img, M, (w, h), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return res, M

    def transform_vertical(self, img, scale=0.3):
        h, w = img.shape[:2]
        src, dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"), \
            np.array([[random.randint(0, int(w*scale)), 0], [random.randint(int(w*(1-scale)), w-1), 0],
                      [w - 1, h - 1], [0, h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        res = cv2.warpPerspective(img, M, (w, h), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return res, M


