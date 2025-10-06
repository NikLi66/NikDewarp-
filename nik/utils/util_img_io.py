import random

import cv2
import numpy as np
import pickle as pk

def save_img(img_path, array):
    print("save img to %s with shape %s" % (img_path, str(array.shape)))
    cv2.imwrite(img_path, array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def dewarp(img, fp):
    return cv2.remap(img, fp[:, :, 0], fp[:, :, 1], interpolation=cv2.INTER_LINEAR).reshape((fp.shape[0], fp.shape[1], img.shape[2]))


def get_img_offset(img_height, img_width):
    img_offset = np.zeros([img_height, img_width, 2], dtype=np.float32)
    for i in range(img_height):
        for j in range(img_width):
            img_offset[i][j][0] = j
            img_offset[i][j][1] = i
    return img_offset


def get_geom_transform(img_height, img_width, type):
    if type == 0:
        # 顺时针90度  (i, j) -> (j, img_width-i)
        img_transform = np.zeros((img_width, img_height, 2), dtype=np.float32)
        fp_transfrom = np.zeros((img_height, img_width, 2), dtype=np.float32)
        for i in range(img_height):
            for j in range(img_width):
                img_transform[j, i, 0] = j  # width
                img_transform[j, i, 1] = img_height - i
                fp_transfrom[i, j, 0] = img_height - i
                fp_transfrom[i, j, 1] = j
        return img_transform, fp_transfrom
    elif type == 1:
        # 顺时针180度  (i, j) -> (j, img_width-i)
        img_transform = np.zeros((img_height, img_width, 2), dtype=np.float32)
        fp_transfrom = np.zeros((img_height, img_width, 2), dtype=np.float32)
        for i in range(img_height):
            for j in range(img_width):
                img_transform[i, j, 0] = j  # width
                img_transform[i, j, 1] = img_height - i
                fp_transfrom[i, j, 0] = j
                fp_transfrom[i, j, 1] = img_height - i
        return img_transform, fp_transfrom
    elif type == 2:
        # 顺时针180度  (i, j) -> (j, img_width-i)
        img_transform = np.zeros((img_width, img_height, 2), dtype=np.float32)
        fp_transfrom = np.zeros((img_height, img_width, 2), dtype=np.float32)
        for i in range(img_height):
            for j in range(img_width):
                img_transform[j, i, 0] = img_width - j  # width
                img_transform[j, i, 1] = i
                fp_transfrom[i, j, 0] = i
                fp_transfrom[i, j, 1] = img_width - j
        return img_transform, fp_transfrom
    else:
        raise Exception("error")


class TalImgTransform(object):
    def __init__(self, img_height, img_width):
        self._img_height = img_height
        self._img_width = img_width
        self._img_offset = get_img_offset(img_height, img_width)
        self._img_transform_0, self._fp_transfrom_0 = get_geom_transform(img_height, img_width, 0)
        self._img_transform_1, self._fp_transfrom_1 = get_geom_transform(img_height, img_width, 1)
        self._img_transform_2, self._fp_transfrom_2 = get_geom_transform(img_height, img_width, 2)

    def process(self, img, fp, type = -1):
        if type not in [0,1,2,3]:
            type = random.randint(0, 3)
        if type == 0:
            img = cv2.remap(img, self._img_transform_0[:, :, 0], self._img_transform_0[:, :, 1],
                            interpolation=cv2.INTER_LINEAR)
            fp = cv2.remap(self._fp_transfrom_0, fp[:, :, 0], fp[:, :, 1],
                           interpolation=cv2.INTER_LINEAR)
            if self._img_height != self._img_width:
                scale3 = self._img_height / self._img_width
                scale4 = self._img_width / self._img_height
                img = cv2.resize(img, (self._img_width, self._img_height), interpolation=cv2.INTER_LINEAR)
                fp[:, :, 0] = fp[:, :, 0] * scale4  # width
                fp[:, :, 1] = fp[:, :, 1] * scale3  # height
            return img, fp
        elif type == 1:
            img = cv2.remap(img, self._img_transform_1[:, :, 0], self._img_transform_1[:, :, 1],
                            interpolation=cv2.INTER_LINEAR)
            fp = cv2.remap(self._fp_transfrom_1, fp[:, :, 0], fp[:, :, 1],
                            interpolation=cv2.INTER_LINEAR)
            return img, fp
        elif type == 2:
            img = cv2.remap(img, self._img_transform_2[:, :, 0], self._img_transform_2[:, :, 1],
                            interpolation=cv2.INTER_LINEAR)
            fp = cv2.remap(self._fp_transfrom_2, fp[:, :, 0], fp[:, :, 1],
                           interpolation=cv2.INTER_LINEAR)
            if self._img_height != self._img_width:
                scale3 = self._img_height / self._img_width
                scale4 = self._img_width / self._img_height
                img = cv2.resize(img, (self._img_width, self._img_height), interpolation=cv2.INTER_LINEAR)
                fp[:, :, 0] = fp[:, :, 0] * scale4  # width
                fp[:, :, 1] = fp[:, :, 1] * scale3  # height
            return img, fp
        else:
            return img, fp


def test_geom_transform():
    img1_path = "/Users/tal/Documents/github/xpad_img/workdir/user.jpg"
    img2_path = "/Users/tal/Documents/github/xpad_img/workdir/user2.jpg"
    warp_path = "/Users/tal/Documents/github/xpad_img/workdir/warp.jpg"

    img1 = cv2.imread(img1_path, 1)

    img_transform, fp_transfrom = get_geom_transform(img1.shape[0], img1.shape[1], 0)
    img2 = cv2.remap(img1, img_transform[:, :, 0], img_transform[:, :, 1], interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(img2_path, img2, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    warp = cv2.remap(img2, fp_transfrom[:, :, 0], fp_transfrom[:, :, 1], interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(warp_path, warp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def test_tal_transform():
    fidu_path = "/Users/tal/Documents/github/xpad_img/workdir/sample/MaurdorF17_55_3_fold.gw"
    with open(fidu_path, 'rb') as f:
        perturbed_data = pk.load(f)

    img = perturbed_data.get('image')  # HWC
    fp = perturbed_data.get('fiducial_points')  # WHC
    img = np.array(img, dtype=np.float32)
    fp = np.array(fp, dtype=np.float32).transpose(1, 0, 2)  # HWC
    fp = cv2.resize(fp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    warp1 = cv2.remap(img, fp[:, :, 0], fp[:, :, 1], interpolation=cv2.INTER_LINEAR)

    tal_transform = TalImgTransform(img_height=img.shape[0], img_width=img.shape[1])
    img2, fp2 = tal_transform.process(img, fp, 2)
    warp2 = cv2.remap(img2, fp2[:, :, 0], fp2[:, :, 1], interpolation=cv2.INTER_LINEAR)

    img1_path = "/Users/tal/Documents/github/xpad_img/workdir/img1.jpg"
    warp1_path = "/Users/tal/Documents/github/xpad_img/workdir/warp1.jpg"

    img2_path = "/Users/tal/Documents/github/xpad_img/workdir/img2.jpg"
    warp2_path = "/Users/tal/Documents/github/xpad_img/workdir/warp2.jpg"

    save_img(img1_path, img)
    save_img(warp1_path, warp1)

    save_img(img2_path, img2)
    save_img(warp2_path, warp2)

    diff = warp1 - warp2
    print(np.average(np.abs(diff)))


if __name__ == '__main__':
    # test_geom_transform()
    test_tal_transform()
