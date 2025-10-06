import albumentations as A
import cv2
import random
import numpy as np
import pickle as pk
from nik.utils.util_zip import zip_decode
from nik.utils.util_img_io import get_img_offset

def read_file(img_path, id):
    with open(img_path, 'rb') as f:
        perturbed_data = pk.load(f)

    img = perturbed_data[id]['img']  # HWC [1024, 960, 3]
    fp = perturbed_data[id]['fp']  # WHC [693, 1012, 2]
    ori = perturbed_data[id]['ori']  # HWC [1012, 693, 3]

    return img, fp, ori

def save_img(img_path, array):
    print(array.shape)
    cv2.imwrite(img_path, array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def get_appearance_transform(transform_types):
    """
    Returns an albumentation compose augmentation.

    transform_type is a list containing types of pixel-wise data augmentation to use.
    Possible augmentations are 'shadow', 'blur', 'visual', 'noise', 'color'.
    """

    transforms = []
    if "shadow" in transform_types:
        transforms.append(A.RandomShadow(p=0.1))
    if "blur" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.Defocus(p=0.5),
                    A.Downscale(p=0.15, interpolation=cv2.INTER_LINEAR),
                    A.GaussianBlur(p=0.65),
                    A.MedianBlur(p=0.15),
                ],
                p=0.75,
            )
        )
    if "visual" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.ToSepia(p=0.15),
                    A.ToGray(p=0.20),
                    A.Equalize(p=0.15),
                    A.Sharpen(p=0.20),
                ],
                p=0.5,
            )
        )
    if "noise" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.GaussNoise(var_limit=(10.0, 20.0), p=0.70),
                    A.ISONoise(intensity=(0.1, 0.25), p=0.30),
                ],
                p=0.6,
            )
        )
    if "color" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.ColorJitter(p=0.5),
                    A.HueSaturationValue(p=0.10),
                    A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.25), p=0.85),
                ],
                p=0.95,
            )
        )

    return A.Compose(transforms=transforms)

def get_geometric_transform(transform_types):
    """
    Returns an albumentation compose augmentation.

    transform_type is a list containing types of geometric data augmentation to use.
    Possible augmentations are 'rotate', 'flip' and 'perspective'.
    """

    transforms = []
    if "rotate" in transform_types:
        transforms.append(
            A.SafeRotate(
                limit=(-30, 30),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            )
        )
    if "flip" in transform_types:
        transforms.append(A.HorizontalFlip(p=0.25))

    if "perspective" in transform_types:
        transforms.append(A.Perspective(p=0.5))

    return A.ReplayCompose(
        transforms=transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

def crop_image_tight(img, grid2D):
    """
    Crops the image tightly around the keypoints in grid2D.
    This function creates a tight crop around the document in the image.
    """
    size = img.shape

    minx = np.floor(np.amin(grid2D[:, :, 0])).astype(int)
    maxx = np.ceil(np.amax(grid2D[:, :, 0])).astype(int)
    miny = np.floor(np.amin(grid2D[ :, :, 1])).astype(int)
    maxy = np.ceil(np.amax(grid2D[:, :, 1])).astype(int)
    s = 20
    s = min(min(s, minx), miny)  # s shouldn't be smaller than actually available natural padding is
    s = min(min(s, size[1] - 1 - maxx), size[0] - 1 - maxy)

    # Crop the image slightly larger than necessary
    img = img[miny - s : maxy + s, minx - s : maxx + s, :]
    cx1 = random.randint(0, max(s - 5, 1))
    cx2 = random.randint(0, max(s - 5, 1)) + 1
    cy1 = random.randint(0, max(s - 5, 1))
    cy2 = random.randint(0, max(s - 5, 1)) + 1

    img = img[cy1:-cy2, cx1:-cx2, :]
    top = miny - s + cy1
    bot = size[0] - maxy - s + cy2
    left = minx - s + cx1
    right = size[1] - maxx - s + cx2
    return img, top, bot, left, right

def crop_tight(img_RGB, grid2D):
    # The incoming grid2D array is expressed in pixel coordinates (resolution of img_RGB before crop/resize)
    size = img_RGB.shape[:2]
    img, top, bot, left, right = crop_image_tight(img_RGB, grid2D)
    # print(img.shape)
    img = cv2.resize(img, size)

    grid2D[:, :, 0] = (grid2D[:, :, 0] - left) * size[1] / (size[1] - left - right)
    grid2D[:, :, 1] = (grid2D[:, :, 1] - top) * size[0] / (size[0] - top - bot)

    return img, grid2D

if __name__ == '__main__':

    appearance_transform = get_appearance_transform(["shadow", "blur", "noise", "visual", "color"])
    geometric_transform = get_geometric_transform(["rotate", "flip", "perspective"])

    pk_path = "/Users/tal/Documents/github/xpad_img/workdir/512/0.pk"
    pk_id = 10
    img_height = 448
    img_width = 448

    offset = get_img_offset(img_height, img_width)

    img, fp, ori = read_file(pk_path, pk_id)

    img = zip_decode(img, np.uint8, [img_height, img_width, 1])
    ori = zip_decode(ori, np.uint8, [img_height, img_width, 1]).astype(np.float32)
    fp = zip_decode(fp, np.int16, [img_height, img_width, 2]).astype(np.float32)
    fp = fp + offset
    warp1_path = "/Users/tal/Documents/github/xpad_img/workdir/trans_warp1.jpg"
    warp1 = cv2.remap(img, fp[:, :, 0], fp[:, :, 1], interpolation=cv2.INTER_LINEAR)
    save_img(warp1_path, warp1)

    # transform
    img = np.concatenate([img,img,img], axis=2)
    img = appearance_transform(image=img)["image"]
    transformed = geometric_transform(
        image=img,
        keypoints=fp.reshape(-1, 2),
    )
    img = transformed["image"]
    fp = np.array(transformed["keypoints"]).reshape(img_height, img_width, 2)

    img = img.astype(np.float32)
    fp = fp.astype(np.float32)

    img,fp = crop_tight(img,fp)

    warp = cv2.remap(img, fp[:, :, 0], fp[:, :, 1], interpolation=cv2.INTER_LINEAR)

    img_path = "/Users/tal/Documents/github/xpad_img/workdir/trans_img.jpg"
    warp_path = "/Users/tal/Documents/github/xpad_img/workdir/trans_warp.jpg"
    ori_path = "/Users/tal/Documents/github/xpad_img/workdir/trans_ori.jpg"
    save_img(img_path, img)
    save_img(warp_path, warp)
    save_img(ori_path, ori)

