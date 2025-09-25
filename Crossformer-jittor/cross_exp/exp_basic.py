import os
import jittor as jt
from jittor import nn, optim
if hasattr(jt, 'has_cuda') and jt.has_cuda:
    jt.flags.use_cuda = 1
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = None
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = None
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    