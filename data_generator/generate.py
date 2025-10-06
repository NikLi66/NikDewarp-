'''
GuoWang xie
set up :2020-1-9
intergrate img and label into one file

-- fiducial1024_v1
'''

import argparse
import sys, os
import pickle
import random
import collections
import json
import numpy as np
import scipy.io as io
import scipy.misc as m
import matplotlib.pyplot as plt
import glob
import math
import time

import threading
import multiprocessing as mp
from multiprocessing import Pool
import re
import cv2
# sys.path.append('/lustre/home/gwxie/hope/project/dewarp/datasets/')	# /lustre/home/gwxie/program/project/unwarp/perturbed_imgaes/GAN
from utils import *


def getDatasets(dir):
    return os.listdir(dir)

def save_img(path, bg_path, fold_curve='fold', repeat_time=4,
             relativeShift_position='relativeShift_v2', save_path=None):
    try:
        origin_img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        save_img_shape = [512 * 2, 480 * 2]  # 320
        reduce_value = np.random.choice([2 * 2, 4 * 2, 8 * 2, 16 * 2, 24 * 2, 32 * 2, 40 * 2, 48 * 2],
                                        p=[0.02, 0.18, 0.2, 0.3, 0.1, 0.1, 0.08, 0.02])
        # reduce_value = 0
        base_img_shrink = save_img_shape[0] - reduce_value

        enlarge_img_shrink = [512 * 4, 480 * 4]  # 420

        im_lr = origin_img.shape[0] #原始前景图的宽和长
        im_ud = origin_img.shape[1]

        # reduce_value_v2 = 0
        reduce_value_v2 = np.random.choice([2 * 2, 4 * 2, 8 * 2, 16 * 2, 24 * 2, 28 * 2, 32 * 2, 48 * 2],
                                           p=[0.02, 0.18, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])

        # resize 前景
        if im_lr > im_ud:
            im_ud = min(int(im_ud / im_lr * base_img_shrink), save_img_shape[1] - reduce_value_v2)
            im_lr = save_img_shape[0] - reduce_value
        else:
            base_img_shrink = save_img_shape[1] - reduce_value
            im_lr = min(int(im_lr / im_ud * base_img_shrink), save_img_shape[0] - reduce_value_v2)
            im_ud = base_img_shrink


        if round(im_lr / im_ud, 2) < 0.5 or round(im_ud / im_lr, 2) < 0.5:
            repeat_time = min(repeat_time, 8)

        edge_padding = random.randint(0, 10)
        fiducial_points = (im_lr-2*edge_padding, im_ud-2*edge_padding)
        im_lr -= (2 * edge_padding)  # im_lr % (fiducial_points-1) - 1
        im_ud -= (2 * edge_padding)  # im_ud % (fiducial_points-1) - 1
        im_hight = np.linspace(edge_padding, im_lr - edge_padding, fiducial_points[0], dtype=np.int64)
        im_wide = np.linspace(edge_padding, im_ud - edge_padding, fiducial_points[1], dtype=np.int64) #np.linspace它用于在指定的区间内创建等间隔的数值
        # im_lr -= im_lr % (fiducial_points-1) - (1+2*edge_padding)		# im_lr % (fiducial_points-1) - 1
        # im_ud -= im_ud % (fiducial_points-1) - (1+2*edge_padding)		# im_ud % (fiducial_points-1) - 1
        # im_hight = np.linspace(edge_padding, im_lr - (1+edge_padding), fiducial_points, dtype=np.int64)
        # im_wide = np.linspace(edge_padding, im_ud - (1+edge_padding), fiducial_points, dtype=np.int64)
        im_x, im_y = np.meshgrid(im_hight, im_wide)
        # np.meshgrid(x, y) 代表的是将x中每一个数据和y中每一个数据组合生成很多点,然后将这些点的x坐标放入到X中,y坐标放入Y中,并且相应位置是对应的
        segment_x = (im_lr) // (fiducial_points[0] - 1)
        segment_y = (im_ud) // (fiducial_points[1] - 1)


        self_origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
        target_img = self_origin_img[:,:,:]

        perturbed_bg_img = cv2.imread(bg_path, flags=cv2.IMREAD_COLOR)

        mesh_shape = self_origin_img.shape[:2]

        self_synthesis_perturbed_img = np.full((enlarge_img_shrink[0], enlarge_img_shrink[1], 3), 256,
                                               dtype=np.float32)  # np.zeros_like(perturbed_bg_img)

        self_new_shape = self_synthesis_perturbed_img.shape[:2]
        perturbed_bg_img = cv2.resize(perturbed_bg_img, (save_img_shape[1], save_img_shape[0]), cv2.INPAINT_TELEA)
        # perturbed_bg_img 背景图变颜色+resize

        # plt.imshow(perturbed_bg_img)
        # plt.show()

        origin_pixel_position = np.argwhere(np.zeros(mesh_shape, dtype=np.uint32) == 0).reshape(mesh_shape[0],
                                                                                                mesh_shape[1], 2)
        pixel_position = np.argwhere(np.zeros(self_new_shape, dtype=np.uint32) == 0).reshape(self_new_shape[0],
                                                                                             self_new_shape[1], 2)
        self_perturbed_xy_ = np.zeros((self_new_shape[0], self_new_shape[1], 2))

        self_synthesis_perturbed_label = np.zeros((self_new_shape[0], self_new_shape[1], 2))
        x_min, y_min, x_max, y_max = self_adjust_position_v2(0, 0, mesh_shape[0], mesh_shape[1], save_img_shape)
        origin_pixel_position += [x_min, y_min]

        x_min, y_min, x_max, y_max = self_adjust_position(0, 0, mesh_shape[0], mesh_shape[1], self_new_shape)
        x_shift = random.randint(-enlarge_img_shrink[0] // 16, enlarge_img_shrink[0] // 16)
        y_shift = random.randint(-enlarge_img_shrink[1] // 16, enlarge_img_shrink[1] // 16)
        x_min += x_shift
        x_max += x_shift
        y_min += y_shift
        y_max += y_shift

        '''im_x,y'''
        im_x += x_min
        im_y += y_min

        self_synthesis_perturbed_img[x_min:x_max, y_min:y_max] = self_origin_img
        self_synthesis_perturbed_label[x_min:x_max, y_min:y_max] = origin_pixel_position

        synthesis_perturbed_img_map = self_synthesis_perturbed_img.copy()
        synthesis_perturbed_label_map = self_synthesis_perturbed_label.copy()

        foreORbackground_label = np.full((mesh_shape), 1, dtype=np.int16)
        foreORbackground_label_map = np.full((self_new_shape), 0, dtype=np.int16)
        foreORbackground_label_map[x_min:x_max, y_min:y_max] = foreORbackground_label

        '''*****************************************************************'''
        is_normalizationFun_mixture = self_is_perform(0.2, 0.8) #随机True or False
        # print("is_normalizationFun_mixture",is_normalizationFun_mixture)

        # if not is_normalizationFun_mixture:
        normalizationFun_0_1 = False

        if fold_curve == 'fold':
            fold_curve_random = True
            # is_normalizationFun_mixture = False
            normalizationFun_0_1 = self_is_perform(0.2, 0.8)
            if is_normalizationFun_mixture:
                alpha_perturbed = random.randint(80, 120) / 100
            else:
                if normalizationFun_0_1 and repeat_time < 8:
                    alpha_perturbed = random.randint(50, 70) / 100
                else:
                    alpha_perturbed = random.randint(70, 130) / 100
        else:
            fold_curve_random = self_is_perform(0.1, 0.9)  #随机True or False # False
            alpha_perturbed = random.randint(80, 160) / 100
        synthesis_perturbed_img = np.full_like(self_synthesis_perturbed_img, 256) #形状与前一个变量完全相同，填充值为256
        synthesis_perturbed_label = np.zeros_like(self_synthesis_perturbed_label)

        alpha_perturbed_change = self_is_perform(0.5, 0.5)
        p_pp_choice = self_is_perform(0.8, 0.2) if fold_curve == 'fold' else self_is_perform(0.1, 0.9)
        for repeat_i in range(repeat_time):

            if alpha_perturbed_change:
                if fold_curve == 'fold':
                    if is_normalizationFun_mixture:
                        alpha_perturbed = random.randint(80, 120) / 100
                    else:
                        if normalizationFun_0_1 and repeat_time < 8:
                            alpha_perturbed = random.randint(50, 70) / 100
                        else:
                            alpha_perturbed = random.randint(70, 130) / 100
                else:
                    alpha_perturbed = random.randint(80, 160) / 100
            ''''''
            linspace_x = [0, (self_new_shape[0] - im_lr) // 2 - 1,
                          self_new_shape[0] - (self_new_shape[0] - im_lr) // 2 - 1, self_new_shape[0] - 1]
            linspace_y = [0, (self_new_shape[1] - im_ud) // 2 - 1,
                          self_new_shape[1] - (self_new_shape[1] - im_ud) // 2 - 1, self_new_shape[1] - 1]
            linspace_x_seq = [1, 2, 3]
            linspace_y_seq = [1, 2, 3]
            r_x = random.choice(linspace_x_seq)
            r_y = random.choice(linspace_y_seq)
            perturbed_p = np.array(
                [random.randint(linspace_x[r_x - 1] * 10, linspace_x[r_x] * 10),
                 random.randint(linspace_y[r_y - 1] * 10, linspace_y[r_y] * 10)]) / 10
            if ((r_x == 1 or r_x == 3) and (r_y == 1 or r_y == 3)) and p_pp_choice:
                linspace_x_seq.remove(r_x)
                linspace_y_seq.remove(r_y)
            r_x = random.choice(linspace_x_seq)
            r_y = random.choice(linspace_y_seq)
            perturbed_pp = np.array(
                [random.randint(linspace_x[r_x - 1] * 10, linspace_x[r_x] * 10),
                 random.randint(linspace_y[r_y - 1] * 10, linspace_y[r_y] * 10)]) / 10

            perturbed_vp = perturbed_pp - perturbed_p
            perturbed_vp_norm = np.linalg.norm(perturbed_vp)

            perturbed_distance_vertex_and_line = np.dot((perturbed_p - pixel_position),
                                                        perturbed_vp) / perturbed_vp_norm
            ''''''
            # perturbed_v = np.array([random.randint(-3000, 3000) / 100, random.randint(-3000, 3000) / 100])
            # perturbed_v = np.array([random.randint(-4000, 4000) / 100, random.randint(-4000, 4000) / 100])
            if fold_curve == 'fold' and self_is_perform(0.6, 0.4):
                # perturbed_v = np.array([random.randint(-9000, 9000) / 100, random.randint(-9000, 9000) / 100])
                perturbed_v = np.array([random.randint(-10000, 10000) / 100, random.randint(-10000, 10000) / 100])
            # perturbed_v = np.array([random.randint(-11000, 11000) / 100, random.randint(-11000, 11000) / 100])
            else:
                # perturbed_v = np.array([random.randint(-9000, 9000) / 100, random.randint(-9000, 9000) / 100])
                # perturbed_v = np.array([random.randint(-16000, 16000) / 100, random.randint(-16000, 16000) / 100])
                perturbed_v = np.array([random.randint(-8000, 8000) / 100, random.randint(-8000, 8000) / 100])
            # perturbed_v = np.array([random.randint(-3500, 3500) / 100, random.randint(-3500, 3500) / 100])
            # perturbed_v = np.array([random.randint(-600, 600) / 10, random.randint(-600, 600) / 10])
            ''''''
            if fold_curve == 'fold':
                if is_normalizationFun_mixture:
                    if self_is_perform(0.5, 0.5):
                        perturbed_d = np.abs(self_get_normalize(perturbed_distance_vertex_and_line))
                    else:
                        perturbed_d = self_get_0_1_d(np.abs(perturbed_distance_vertex_and_line), random.randint(1, 2))
                else:
                    if normalizationFun_0_1:
                        perturbed_d = self_get_0_1_d(np.abs(perturbed_distance_vertex_and_line), 2)
                    else:
                        perturbed_d = np.abs(self_get_normalize(perturbed_distance_vertex_and_line))

            else:
                if is_normalizationFun_mixture:
                    if self_is_perform(0.5, 0.5):
                        perturbed_d = np.abs(self_get_normalize(perturbed_distance_vertex_and_line))
                    else:
                        perturbed_d = self_get_0_1_d(np.abs(perturbed_distance_vertex_and_line), random.randint(1, 2))
                else:
                    if normalizationFun_0_1:
                        perturbed_d = self_get_0_1_d(np.abs(perturbed_distance_vertex_and_line), 2)
                    else:
                        perturbed_d = np.abs(self_get_normalize(perturbed_distance_vertex_and_line))
            ''''''
            if fold_curve_random:
                # omega_perturbed = (alpha_perturbed+0.2) / (perturbed_d + alpha_perturbed)
                # omega_perturbed = alpha_perturbed**perturbed_d
                omega_perturbed = alpha_perturbed / (perturbed_d + alpha_perturbed)
            else:
                omega_perturbed = 1 - perturbed_d ** alpha_perturbed

            '''shadow'''
            if self_is_perform(0.6, 0.4):
                synthesis_perturbed_img_map[x_min:x_max, y_min:y_max] = np.minimum(np.maximum(
                    synthesis_perturbed_img_map[x_min:x_max, y_min:y_max] - np.int16(np.round(
                        omega_perturbed[x_min:x_max, y_min:y_max].repeat(3).reshape(x_max - x_min, y_max - y_min,
                                                                                    3) * abs(
                            np.linalg.norm(perturbed_v // 2)) * np.array(
                            [0.4 - random.random() * 0.1, 0.4 - random.random() * 0.1, 0.4 - random.random() * 0.1]))),
                    0), 255)
            ''''''

            if relativeShift_position in ['position', 'relativeShift_v2']:
                self_perturbed_xy_ += np.array(
                    [omega_perturbed * perturbed_v[0], omega_perturbed * perturbed_v[1]]).transpose(1, 2, 0)
            else:
                print('relativeShift_position error')
                exit()

        '''perspective'''

        # perspective_shreshold = random.randint(26, 36) * 10  # 280
        perspective_shreshold = 0
        x_min_per, y_min_per, x_max_per, y_max_per = self_adjust_position(perspective_shreshold, perspective_shreshold,
                                                                          self_new_shape[0] - perspective_shreshold,
                                                                          self_new_shape[1] - perspective_shreshold, self_new_shape)
        pts1 = np.float32(
            [[x_min_per, y_min_per], [x_max_per, y_min_per], [x_min_per, y_max_per], [x_max_per, y_max_per]])
        e_1_ = x_max_per - x_min_per
        e_2_ = y_max_per - y_min_per
        e_3_ = e_2_
        e_4_ = e_1_
        perspective_shreshold_h = e_1_ * 0.02
        perspective_shreshold_w = e_2_ * 0.02
        a_min_, a_max_ = 70, 110
        # if self.is_perform(1, 0):
        if fold_curve == 'curve' and self_is_perform(0.5, 0.5):
            if self_is_perform(0.5, 0.5):
                while True:
                    pts2 = np.around(
                        np.float32([[x_min_per - (random.random()) * perspective_shreshold,
                                     y_min_per + (random.random()) * perspective_shreshold],
                                    [x_max_per - (random.random()) * perspective_shreshold,
                                     y_min_per - (random.random()) * perspective_shreshold],
                                    [x_min_per + (random.random()) * perspective_shreshold,
                                     y_max_per + (random.random()) * perspective_shreshold],
                                    [x_max_per + (random.random()) * perspective_shreshold,
                                     y_max_per - (random.random()) * perspective_shreshold]]))  # right
                    e_1 = np.linalg.norm(pts2[0] - pts2[1])
                    e_2 = np.linalg.norm(pts2[0] - pts2[2])
                    e_3 = np.linalg.norm(pts2[1] - pts2[3])
                    e_4 = np.linalg.norm(pts2[2] - pts2[3])
                    if e_1_ + perspective_shreshold_h > e_1 and e_2_ + perspective_shreshold_w > e_2 and e_3_ + perspective_shreshold_w > e_3 and e_4_ + perspective_shreshold_h > e_4 and \
                            e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
                            abs(e_1 - e_4) < perspective_shreshold_h and abs(e_2 - e_3) < perspective_shreshold_w:
                        a0_, a1_, a2_, a3_ = self_get_angle_4(pts2)
                        if (a0_ > a_min_ and a0_ < a_max_) or (a1_ > a_min_ and a1_ < a_max_) or (
                                a2_ > a_min_ and a2_ < a_max_) or (a3_ > a_min_ and a3_ < a_max_):
                            break
            else:
                while True:
                    pts2 = np.around(
                        np.float32([[x_min_per + (random.random()) * perspective_shreshold,
                                     y_min_per - (random.random()) * perspective_shreshold],
                                    [x_max_per + (random.random()) * perspective_shreshold,
                                     y_min_per + (random.random()) * perspective_shreshold],
                                    [x_min_per - (random.random()) * perspective_shreshold,
                                     y_max_per - (random.random()) * perspective_shreshold],
                                    [x_max_per - (random.random()) * perspective_shreshold,
                                     y_max_per + (random.random()) * perspective_shreshold]]))
                    e_1 = np.linalg.norm(pts2[0] - pts2[1])
                    e_2 = np.linalg.norm(pts2[0] - pts2[2])
                    e_3 = np.linalg.norm(pts2[1] - pts2[3])
                    e_4 = np.linalg.norm(pts2[2] - pts2[3])
                    if e_1_ + perspective_shreshold_h > e_1 and e_2_ + perspective_shreshold_w > e_2 and e_3_ + perspective_shreshold_w > e_3 and e_4_ + perspective_shreshold_h > e_4 and \
                            e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
                            abs(e_1 - e_4) < perspective_shreshold_h and abs(e_2 - e_3) < perspective_shreshold_w:
                        a0_, a1_, a2_, a3_ = self_get_angle_4(pts2)
                        if (a0_ > a_min_ and a0_ < a_max_) or (a1_ > a_min_ and a1_ < a_max_) or (
                                a2_ > a_min_ and a2_ < a_max_) or (a3_ > a_min_ and a3_ < a_max_):
                            break

        else:
            while True:
                pts2 = np.around(np.float32([[x_min_per + (random.random() - 0.5) * perspective_shreshold,
                                              y_min_per + (random.random() - 0.5) * perspective_shreshold],
                                             [x_max_per + (random.random() - 0.5) * perspective_shreshold,
                                              y_min_per + (random.random() - 0.5) * perspective_shreshold],
                                             [x_min_per + (random.random() - 0.5) * perspective_shreshold,
                                              y_max_per + (random.random() - 0.5) * perspective_shreshold],
                                             [x_max_per + (random.random() - 0.5) * perspective_shreshold,
                                              y_max_per + (random.random() - 0.5) * perspective_shreshold]]))
                e_1 = np.linalg.norm(pts2[0] - pts2[1])
                e_2 = np.linalg.norm(pts2[0] - pts2[2])
                e_3 = np.linalg.norm(pts2[1] - pts2[3])
                e_4 = np.linalg.norm(pts2[2] - pts2[3])
                if e_1_ + perspective_shreshold_h > e_1 and e_2_ + perspective_shreshold_w > e_2 and e_3_ + perspective_shreshold_w > e_3 and e_4_ + perspective_shreshold_h > e_4 and \
                        e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
                        abs(e_1 - e_4) < perspective_shreshold_h and abs(e_2 - e_3) < perspective_shreshold_w:
                    a0_, a1_, a2_, a3_ = self_get_angle_4(pts2)
                    if (a0_ > a_min_ and a0_ < a_max_) or (a1_ > a_min_ and a1_ < a_max_) or (
                            a2_ > a_min_ and a2_ < a_max_) or (a3_ > a_min_ and a3_ < a_max_):
                        break

        M = cv2.getPerspectiveTransform(pts1, pts2)
        one = np.ones((self_new_shape[0], self_new_shape[1], 1), dtype=np.int16)
        matr = np.dstack((pixel_position, one))
        new = np.dot(M, matr.reshape(-1, 3).T).T.reshape(self_new_shape[0], self_new_shape[1], 3)
        x = new[:, :, 0] / new[:, :, 2]
        y = new[:, :, 1] / new[:, :, 2]
        perturbed_xy_ = np.dstack((x, y))
        # perturbed_xy_round_int = np.around(cv2.bilateralFilter(perturbed_xy_round_int, 9, 75, 75))
        # perturbed_xy_round_int = np.around(cv2.blur(perturbed_xy_, (17, 17)))
        # perturbed_xy_round_int = cv2.blur(perturbed_xy_round_int, (17, 17))
        # perturbed_xy_round_int = cv2.GaussianBlur(perturbed_xy_round_int, (7, 7), 0)
        perturbed_xy_ = perturbed_xy_ - np.min(perturbed_xy_.T.reshape(2, -1), 1)
        # perturbed_xy_round_int = np.around(perturbed_xy_round_int-np.min(perturbed_xy_round_int.T.reshape(2, -1), 1)).astype(np.int16)

        self_perturbed_xy_ += perturbed_xy_

        '''perspective end'''

        '''to img'''
        flat_position = np.argwhere(np.zeros(self_new_shape, dtype=np.uint32) == 0).reshape(
            self_new_shape[0] * self_new_shape[1], 2)
        # self.perturbed_xy_ = cv2.blur(self.perturbed_xy_, (7, 7))
        self_perturbed_xy_ = cv2.GaussianBlur(self_perturbed_xy_, (7, 7), 0)

        '''get fiducial points'''
        fiducial_points_coordinate = self_perturbed_xy_[im_x, im_y]

        vtx, wts = self_interp_weights(self_perturbed_xy_.reshape(self_new_shape[0] * self_new_shape[1], 2),
                                       flat_position)
        wts_sum = np.abs(wts).sum(-1)

        # flat_img.reshape(flat_shape[0] * flat_shape[1], 3)[:] = interpolate(pixel, vtx, wts)
        wts = wts[wts_sum <= 1, :]
        vtx = vtx[wts_sum <= 1, :]
        synthesis_perturbed_img.reshape(self_new_shape[0] * self_new_shape[1], 3)[wts_sum <= 1,
        :] = self_interpolate(synthesis_perturbed_img_map.reshape(self_new_shape[0] * self_new_shape[1], 3), vtx, wts)

        synthesis_perturbed_label.reshape(self_new_shape[0] * self_new_shape[1], 2)[wts_sum <= 1,
        :] = self_interpolate(synthesis_perturbed_label_map.reshape(self_new_shape[0] * self_new_shape[1], 2), vtx, wts)

        foreORbackground_label = np.zeros(self_new_shape)
        foreORbackground_label.reshape(self_new_shape[0] * self_new_shape[1], 1)[wts_sum <= 1, :] = self_interpolate(
            foreORbackground_label_map.reshape(self_new_shape[0] * self_new_shape[1], 1), vtx, wts)
        foreORbackground_label[foreORbackground_label < 0.99] = 0
        foreORbackground_label[foreORbackground_label >= 0.99] = 1

        self_synthesis_perturbed_img = synthesis_perturbed_img
        self_synthesis_perturbed_label = synthesis_perturbed_label
        self_foreORbackground_label = foreORbackground_label

        '''clip'''
        perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max = -1, -1, self_new_shape[0], self_new_shape[
            1]
        for x in range(self_new_shape[0] // 2, perturbed_x_max):
            if np.sum(self_synthesis_perturbed_img[x, :]) == 768 * self_new_shape[1] and perturbed_x_max - 1 > x:
                perturbed_x_max = x
                break
        for x in range(self_new_shape[0] // 2, perturbed_x_min, -1):
            if np.sum(self_synthesis_perturbed_img[x, :]) == 768 * self_new_shape[1] and x > 0:
                perturbed_x_min = x
                break
        for y in range(self_new_shape[1] // 2, perturbed_y_max):
            if np.sum(self_synthesis_perturbed_img[:, y]) == 768 * self_new_shape[0] and perturbed_y_max - 1 > y:
                perturbed_y_max = y
                break
        for y in range(self_new_shape[1] // 2, perturbed_y_min, -1):
            if np.sum(self_synthesis_perturbed_img[:, y]) == 768 * self_new_shape[0] and y > 0:
                perturbed_y_min = y
                break

        if perturbed_x_min == 0 or perturbed_x_max == self_new_shape[0] or perturbed_y_min == self_new_shape[
            1] or perturbed_y_max == self_new_shape[1]:
            raise Exception('clip error')

        if perturbed_x_max - perturbed_x_min < im_lr // 2 or perturbed_y_max - perturbed_y_min < im_ud // 2:
            raise Exception('clip error')

        # perfix_ = self_save_suffix + '_' + str(m) + '_' + str(n)  # 注 此处为命名


        is_shrink = False
        if perturbed_x_max - perturbed_x_min > save_img_shape[0] or perturbed_y_max - perturbed_y_min > save_img_shape[
            1]:
            is_shrink = True
            synthesis_perturbed_img = cv2.resize(
                self_synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max,
                :].copy(), (im_ud, im_lr), interpolation=cv2.INTER_LINEAR)
            synthesis_perturbed_label = cv2.resize(
                self_synthesis_perturbed_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max,
                :].copy(), (im_ud, im_lr), interpolation=cv2.INTER_LINEAR)
            foreORbackground_label = cv2.resize(
                self_foreORbackground_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max].copy(),
                (im_ud, im_lr), interpolation=cv2.INTER_LINEAR)
            foreORbackground_label[foreORbackground_label < 0.99] = 0
            foreORbackground_label[foreORbackground_label >= 0.99] = 1
            '''shrink fiducial points'''
            center_x_l, center_y_l = perturbed_x_min + (perturbed_x_max - perturbed_x_min) // 2, perturbed_y_min + (
                        perturbed_y_max - perturbed_y_min) // 2
            shrink_x = im_lr / (perturbed_x_max - perturbed_x_min)
            shrink_y = im_ud / (perturbed_y_max - perturbed_y_min)
            fiducial_points_coordinate *= [shrink_x, shrink_y]
            center_x_l *= shrink_x
            center_y_l *= shrink_y

            perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max = self_adjust_position_v2(0, 0, im_lr,
                                                                                                         im_ud,
                                                                                                         self_new_shape)

            self_synthesis_perturbed_img = np.full_like(self_synthesis_perturbed_img, 256)
            self_synthesis_perturbed_label = np.zeros_like(self_synthesis_perturbed_label)
            self_foreORbackground_label = np.zeros_like(self_foreORbackground_label)
            self_synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max,
            :] = synthesis_perturbed_img
            self_synthesis_perturbed_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max,
            :] = synthesis_perturbed_label
            self_foreORbackground_label[perturbed_x_min:perturbed_x_max,
            perturbed_y_min:perturbed_y_max] = foreORbackground_label

        center_x, center_y = perturbed_x_min + (perturbed_x_max - perturbed_x_min) // 2, perturbed_y_min + (
                    perturbed_y_max - perturbed_y_min) // 2
        if is_shrink:
            fiducial_points_coordinate += [center_x - center_x_l, center_y - center_y_l]

        self_new_shape = save_img_shape
        self_synthesis_perturbed_img = self_synthesis_perturbed_img[
                                       center_x - self_new_shape[0] // 2:center_x + self_new_shape[0] // 2,
                                       center_y - self_new_shape[1] // 2:center_y + self_new_shape[1] // 2,
                                       :].copy()
        self_synthesis_perturbed_label = self_synthesis_perturbed_label[
                                         center_x - self_new_shape[0] // 2:center_x + self_new_shape[0] // 2,
                                         center_y - self_new_shape[1] // 2:center_y + self_new_shape[1] // 2,
                                         :].copy()
        self_foreORbackground_label = self_foreORbackground_label[
                                      center_x - self_new_shape[0] // 2:center_x + self_new_shape[0] // 2,
                                      center_y - self_new_shape[1] // 2:center_y + self_new_shape[1] // 2].copy()

        perturbed_x_ = max(self_new_shape[0] - (perturbed_x_max - perturbed_x_min), 0)
        perturbed_x_min = perturbed_x_ // 2
        perturbed_x_max = self_new_shape[0] - perturbed_x_ // 2 if perturbed_x_ % 2 == 0 else self_new_shape[0] - (
                    perturbed_x_ // 2 + 1)

        perturbed_y_ = max(self_new_shape[1] - (perturbed_y_max - perturbed_y_min), 0)
        perturbed_y_min = perturbed_y_ // 2
        perturbed_y_max = self_new_shape[1] - perturbed_y_ // 2 if perturbed_y_ % 2 == 0 else self_new_shape[1] - (
                    perturbed_y_ // 2 + 1)


        '''save'''
        pixel_position = np.argwhere(np.zeros(self_new_shape, dtype=np.uint32) == 0).reshape(self_new_shape[0],
                                                                                             self_new_shape[1], 2)

        if relativeShift_position == 'relativeShift_v2':
            self_synthesis_perturbed_label -= pixel_position
            fiducial_points_coordinate -= [center_x - self_new_shape[0] // 2, center_y - self_new_shape[1] // 2]

        self_synthesis_perturbed_label[:, :, 0] *= self_foreORbackground_label
        self_synthesis_perturbed_label[:, :, 1] *= self_foreORbackground_label
        self_synthesis_perturbed_img[:, :, 0] *= self_foreORbackground_label
        self_synthesis_perturbed_img[:, :, 1] *= self_foreORbackground_label
        self_synthesis_perturbed_img[:, :, 2] *= self_foreORbackground_label


        '''HSV_v2'''
        perturbed_bg_img = perturbed_bg_img.astype(np.float32)
        if self_is_perform(0.1, 0.9):
            if self_is_perform(0.2, 0.8):
                synthesis_perturbed_img_clip_HSV = self_synthesis_perturbed_img.copy()

                synthesis_perturbed_img_clip_HSV = self_HSV_v1(synthesis_perturbed_img_clip_HSV)

                perturbed_bg_img[:, :, 0] *= 1 - self_foreORbackground_label
                perturbed_bg_img[:, :, 1] *= 1 - self_foreORbackground_label
                perturbed_bg_img[:, :, 2] *= 1 - self_foreORbackground_label

                synthesis_perturbed_img_clip_HSV += perturbed_bg_img
                self_synthesis_perturbed_img = synthesis_perturbed_img_clip_HSV
            else:
                perturbed_bg_img_HSV = perturbed_bg_img
                perturbed_bg_img_HSV = self_HSV_v1(perturbed_bg_img_HSV)

                perturbed_bg_img_HSV[:, :, 0] *= 1 - self_foreORbackground_label
                perturbed_bg_img_HSV[:, :, 1] *= 1 - self_foreORbackground_label
                perturbed_bg_img_HSV[:, :, 2] *= 1 - self_foreORbackground_label

                self_synthesis_perturbed_img += perturbed_bg_img_HSV
            # self.synthesis_perturbed_img[np.sum(self.synthesis_perturbed_img, 2) == 771] = perturbed_bg_img_HSV[np.sum(self.synthesis_perturbed_img, 2) == 771]

        else:
            synthesis_perturbed_img_clip_HSV = self_synthesis_perturbed_img.copy()
            perturbed_bg_img[:, :, 0] *= 1 - self_foreORbackground_label
            perturbed_bg_img[:, :, 1] *= 1 - self_foreORbackground_label
            perturbed_bg_img[:, :, 2] *= 1 - self_foreORbackground_label

            synthesis_perturbed_img_clip_HSV += perturbed_bg_img

            synthesis_perturbed_img_clip_HSV = self_HSV_v1(synthesis_perturbed_img_clip_HSV)

            self_synthesis_perturbed_img = synthesis_perturbed_img_clip_HSV

        ''''''
        # cv2.imwrite(self_save_path+'clip/'+perfix_+'_'+fold_curve+str(perturbed_time)+'-'+str(repeat_time)+'.png', synthesis_perturbed_img_clip)

        self_synthesis_perturbed_img[self_synthesis_perturbed_img < 0] = 0
        self_synthesis_perturbed_img[self_synthesis_perturbed_img > 255] = 255
        self_synthesis_perturbed_img = np.around(self_synthesis_perturbed_img).astype(np.uint8)
        label = np.zeros_like(self_synthesis_perturbed_img, dtype=np.float32)
        label[:, :, :2] = self_synthesis_perturbed_label
        label[:, :, 2] = self_foreORbackground_label

        synthesis_perturbed_color = np.concatenate((self_synthesis_perturbed_img, label), axis=2)
        #这个是要保存的最终图片！包括gw和png 都是这个变量保存的！！

        self_synthesis_perturbed_color = np.zeros_like(synthesis_perturbed_color, dtype=np.float32)
        reduce_value_x = int(round(
            min((random.random() / 2) * (self_new_shape[0] - (perturbed_x_max - perturbed_x_min)),
                min(reduce_value, reduce_value_v2))))
        reduce_value_y = int(round(
            min((random.random() / 2) * (self_new_shape[1] - (perturbed_y_max - perturbed_y_min)),
                min(reduce_value, reduce_value_v2))))
        perturbed_x_min = max(perturbed_x_min - reduce_value_x, 0)
        perturbed_x_max = min(perturbed_x_max + reduce_value_x, self_new_shape[0])
        perturbed_y_min = max(perturbed_y_min - reduce_value_y, 0)
        perturbed_y_max = min(perturbed_y_max + reduce_value_y, self_new_shape[1])

        if im_lr >= im_ud:
            self_synthesis_perturbed_color[:, perturbed_y_min:perturbed_y_max, :] = synthesis_perturbed_color[:,
                                                                                    perturbed_y_min:perturbed_y_max, :]
        # self.synthesis_perturbed_grey[:, perturbed_y_min:perturbed_y_max, :] = synthesis_perturbed_grey[:, perturbed_y_min:perturbed_y_max, :]
        else:
            self_synthesis_perturbed_color[perturbed_x_min:perturbed_x_max, :, :] = synthesis_perturbed_color[
                                                                                    perturbed_x_min:perturbed_x_max, :,
                                                                                    :]
        # self.synthesis_perturbed_grey[perturbed_x_min:perturbed_x_max, :, :] = synthesis_perturbed_grey[perturbed_x_min:perturbed_x_max, :, :]

        '''blur'''#对最终的图片随机加入高斯噪声 cv2.GaussianBlur 它使用高斯滤波器对输入图像进行模糊处理,以减少图像中的噪声和细节
        if self_is_perform(0.1, 0.9):  #随机生成 true or false
            synthesis_perturbed_img_filter = self_synthesis_perturbed_color[:, :, :3].copy()
            if self_is_perform(0.1, 0.9):
                synthesis_perturbed_img_filter = cv2.GaussianBlur(synthesis_perturbed_img_filter, (5, 5), 0)
            else:
                synthesis_perturbed_img_filter = cv2.GaussianBlur(synthesis_perturbed_img_filter, (3, 3), 0)
            if self_is_perform(0.5, 0.5):
                self_synthesis_perturbed_color[:, :, :3][self_synthesis_perturbed_color[:, :, 5] == 1] = \
                synthesis_perturbed_img_filter[self_synthesis_perturbed_color[:, :, 5] == 1]
            else:
                self_synthesis_perturbed_color[:, :, :3] = synthesis_perturbed_img_filter

        # synthesis_perturbed_color 是最终要保存的图片


        fiducial_points_coordinate = fiducial_points_coordinate[:, :, ::-1]
        '''draw fiducial points'''
        # stepSize = 0
        '''forward-begin'''

        '''forward-end'''
        synthesis_perturbed_data = {
            'image': self_synthesis_perturbed_color[:, :, :3].astype(np.uint8),
            'fiducial_points': fiducial_points_coordinate.astype(np.short),
            'segment': np.array((segment_x, segment_y), dtype=np.short),
            'origin': target_img.astype(np.uint8)
        }
        if save_path is not None:
            with open(save_path, 'wb') as f:  #保存gw
                pickle_perturbed_data = pickle.dumps(synthesis_perturbed_data)
                f.write(pickle_perturbed_data)

    except Exception as e:
        print('error:', e)
        synthesis_perturbed_data = None


