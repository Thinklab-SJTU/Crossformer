import argparse
import os
import torch
import pickle

from cross_exp.exp_crossformer import Exp_crossformer
from utils.tools import load_args, string_split

parser = argparse.ArgumentParser(description='CrossFormer')

parser.add_argument('--checkpoint_root', type=str, default='./checkpoints', help='location of the trained model')
parser.add_argument('--setting_name', type=str, default='Crossformer_ETTh1_il168_ol24_sl6_win2_fa10_dm256_nh4_el3_itr0', help='name of the experiment')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

parser.add_argument('--different_split', action='store_true', help='use data split different from training process', default=False)
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2', help='data split of train, vali, test')

parser.add_argument('--inverse', action='store_true', help='inverse output data into the original scale', default=False)
parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.use_multi_gpu = False

args.checkpoint_dir = os.path.join(args.checkpoint_root, args.setting_name)
hyper_parameters = load_args(os.path.join(args.checkpoint_dir, 'args.json'))

#load the pre-trained model
args.data_dim = hyper_parameters['data_dim']; args.in_len = hyper_parameters['in_len']; args.out_len = hyper_parameters['out_len'];
args.seg_len = hyper_parameters['seg_len']; args.win_size = hyper_parameters['win_size']; args.factor = hyper_parameters['factor'];
args.d_model = hyper_parameters['d_model']; args.d_ff = hyper_parameters['d_ff']; args.n_heads = hyper_parameters['n_heads'];
args.e_layers = hyper_parameters['e_layers']; args.dropout = hyper_parameters['dropout']; args.baseline = hyper_parameters['baseline'];
exp = Exp_crossformer(args)
model_dict = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'), map_location='cpu')
exp.model.load_state_dict(model_dict)

#load the data
args.scale_statistic = pickle.load(open(os.path.join(args.checkpoint_dir, 'scale_statistic.pkl'), 'rb'))
args.root_path = hyper_parameters['root_path']; args.data_path = hyper_parameters['data_path'];
if args.different_split:
    data_split = string_split(args.data_split)
    args.data_split = data_split
else:
    args.data_split = hyper_parameters['data_split']

mae, mse, rmse, mape, mspe = exp.eval(args.setting_name, args.save_pred, args.inverse)

folder_path = './results/' + args.setting_name +'/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
log_file = open(folder_path+'metric.log', 'w')
log_file.write('Data Path: {}\n'.format(os.path.join(args.root_path, args.data_path)))
log_file.write('Data Split: {}\n'.format(args.data_split))
log_file.write('Input Length:{}   Output Length:{}\n'.format(args.in_len, args.out_len))
log_file.write('Inverse to original scale: {}\n\n'.format(args.inverse))
log_file.write('MAE:{}\nMSE:{}\nRMSE:{}\nMAPE:{}\nMSPE:{}\n'.format(mae, mse, rmse, mape, mspe))
log_file.close()

