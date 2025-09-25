import numpy as np
import jittor as jt
import json

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj == 'type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        # Jittor 优化器没有 PyTorch 的 param_groups，这里直接改 optimizer.lr
        optimizer.lr = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save(path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        # 如果 data 是 Jittor Var，则把 mean/std 转成相同 dtype 的 jt.Var
        if isinstance(data, jt.Var):
            mean = jt.array(self.mean).astype(data.dtype)
            std = jt.array(self.std).astype(data.dtype)
        else:
            mean = self.mean
            std = self.std
        return (data - mean) / (std + 1e-8)  # 避免除零

    def inverse_transform(self, data):
        if isinstance(data, jt.Var):
            mean = jt.array(self.mean).astype(data.dtype)
            std = jt.array(self.std).astype(data.dtype)
        else:
            mean = self.mean
            std = self.std
        return data * (std + 1e-8) + mean


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]
    return value_list
