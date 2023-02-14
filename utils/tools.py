import numpy as np
import torch
import json

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj=='type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
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
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list
