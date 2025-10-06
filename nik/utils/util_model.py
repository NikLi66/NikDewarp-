import torch
from torch.utils import data
from torch.autograd import Variable, Function
import numpy as np
import sys, os, math
from  collections import OrderedDict

def save_checkpoint(checkpoint_path, model, epoch):
    state = {'state_dict': model.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    fix_state = fix_state_dict(state['state_dict'])
    pretrained_dict = {k: v for k, v in fix_state.items() if k in model_dict}
    model.load_state_dict(pretrained_dict)
    # optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    print('model loaded from %s' % checkpoint_path)
    return start_epoch

def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 去掉 'model.' 前缀
        # name = "model." + k  # 去掉 'model.' 前缀
        new_state_dict[name] = v
    return new_state_dict