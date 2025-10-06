# with open(self.save_path + 'color/' + perfix_ + '_' + fold_curve + '.gw', 'wb') as f:  # 保存gw
#     pickle_perturbed_data = pickle.dumps(synthesis_perturbed_data)
#     f.write(pickle_perturbed_data)
from generate import save_img
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import random
import pickle
import numpy as np
import json
import argparse
from tqdm import tqdm


def main(args):
    img_root_path = './imgs/'
    bg_root_path = './background/'
    output_path = './dewarp/large_dataset/data/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    idx = args.index
    save_path = os.path.join(output_path, idx)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    names = set(os.listdir(save_path))
    img_list = os.listdir(img_root_path)

    bg_list = os.listdir(bg_root_path)

    print('The number of foreground images is:', len(img_list))
    print('The number of background images is:', len(bg_list))
    executor = ProcessPoolExecutor(100)

    print("The data generation is starting...")
    for i, img_name in enumerate(img_list):
        gw_name = img_name.split('.')[0] + '.gw'
        if gw_name in names:
            continue
        img_path = img_root_path + img_name
        bg_name = random.choice(bg_list)
        bg_path = bg_root_path + bg_name
        type = random.choice(['fold', 'curve'])
        repeat_time = min(max(round(np.random.normal(12, 4)), 1), 18)
        path = f"{save_path}/{img_name.split('.')[0]}.gw"
        future = executor.submit(save_img, img_path, bg_path, type, repeat_time, 'relativeShift_v2', path)
    executor.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='index')
    args = parser.parse_args()

    main(args)