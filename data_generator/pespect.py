import cv2
import numpy as np
import pickle
import os
import random


def random_perspective_transform(img):
	"""
	Apply random perspective transform to the input image.
	:param img: input image
	:param max_scale: maximum scaling factor
	:param max_angle: maximum rotation angle in degrees
	:param max_shear: maximum shear factor
	:param max_translation: maximum translation factor
	:return: transformed image
	"""
	# Random rotation
	angle = random.uniform(-10, 10)
	rows, cols = img.shape[:2]
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
	img = cv2.warpAffine(img, M, (cols, rows))

	 

	return img


def self_HSV_v1(synthesis_perturbed_img_clip_HSV):
	synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_RGB2HSV)
	img_h = synthesis_perturbed_img_clip_HSV[:, :, 0].copy()
	# img_s = synthesis_perturbed_img_clip_HSV[:, :, 1].copy()
	img_v = synthesis_perturbed_img_clip_HSV[:, :, 2].copy()

	if 0.2 < random.random() < 0.8:
		img_h = (img_h + (random.random()-0.5) * 360) % 360  # img_h = np.minimum(np.maximum(img_h+20, 0), 360)
	else:
		img_h = (img_h + (random.random()-0.5) * 40) % 360
	# img_s = np.minimum(np.maximum(img_s-0.2, 0), 1)
	img_v = np.minimum(np.maximum(img_v + (random.random()-0.5)*60, 0), 255)
	# img_v = cv2.equalizeHist(img_v.astype(np.uint8))
	synthesis_perturbed_img_clip_HSV[:, :, 0] = img_h
	# synthesis_perturbed_img_clip_HSV[:, :, 1] = img_s
	synthesis_perturbed_img_clip_HSV[:, :, 2] = img_v

	synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_HSV2RGB)

	return synthesis_perturbed_img_clip_HSV

back_path = './background/'
fore_path = './new/'

back_list = os.listdir(back_path)
fore_list = os.listdir(fore_path)

fore_name = random.choice(fore_list)
img = cv2.imread(fore_path + fore_name)
img = random_perspective_transform(img)
cv2.imwrite('fore.jpg', img)

