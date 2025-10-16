import os
import torch
from copy import deepcopy
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import zipfile
import torchvision.models as models
from openstl.models.mfwpn import MFWPN_Model
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
from matplotlib.pyplot import MultipleLocator
from utils.data_sliding import *
import pywt
import pywt.data
import torch.nn.functional as F
from utils import SSIM
from thop import profile

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):

        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(35)
        model_kwargs = dict(
            turbine_coords=getattr(configs, 'turbine_coords', None),
            power_roi_size=getattr(configs, 'power_roi_size', 5),
            power_conv_channels=getattr(configs, 'power_conv_channels', 32),
            power_mlp_hidden=getattr(configs, 'power_mlp_hidden', 64),
            feature_hw=getattr(configs, 'feature_hw', (64, 80))
        )
        self.network = MFWPN_Model(**model_kwargs).to(configs.device)
        adam = torch.optim.Adam([{'params': self.network.parameters()}], lr=0, weight_decay=configs.weight_decay)
        factor = math.sqrt(configs.d_model*configs.warmup)*0.001
        self.opt = NoamOpt(configs.d_model, factor, warmup=configs.warmup, optimizer=adam)
        self.u, self.v = 'u', 'v'
        self.power_loss_fn = nn.MSELoss()

    def loss(self, y_pred, y_true, idx):
        if idx == 'u':
            idx = 0
        if idx == 'v':
            idx = 1
            
        rmse = torch.mean((y_pred[:, :, idx] - y_true[:, :, idx])**2, dim=[2, 3])
        rmse = torch.mean(torch.sqrt(rmse.mean(dim=0)))
            
        return rmse

    def test(self, dataloader_test, ele):
        uv_pred = []
        power_pred = []
        with torch.no_grad():
            for batch in dataloader_test:
                if len(batch) == 2:
                    input_uv, _ = batch
                else:
                    input_uv, _, _ = batch
                outputs = self.network(input_uv.float().to(self.device), ele.float().to(self.device))
                uv_pred.append(outputs['wind'])
                power_output = outputs.get('power')
                if power_output is not None:
                    power_pred.append(power_output)

        uv_tensor = torch.cat(uv_pred, dim=0)
        power_tensor = torch.cat(power_pred, dim=0) if power_pred else None
        return uv_tensor, power_tensor

    def infer(self, dataset, dataloader, ele):
        self.network.eval()
        with torch.no_grad():
            uv_pred, power_pred = self.test(dataloader, ele)
            uv_true = torch.from_numpy(dataset.target).float().to(self.device)

            uv_pred_np = uv_pred
            uv_true_np = uv_true
            
            uv_pred_test = uv_pred_np.to('cpu')
            uv_true_test = uv_true_np.to('cpu')
            
            uv_pred_test = uv_pred_test.numpy()
            uv_true_test = uv_true_test.numpy()
            
            np.save(file='result/uv_pred', arr=uv_pred_test)
            np.save(file='result/uv_true', arr=uv_true_test)

            loss_u = self.loss(uv_pred, uv_true, self.u).item()
            loss_v = self.loss(uv_pred, uv_true, self.v).item()
            loss_power = None
            if power_pred is not None and getattr(dataset, 'target_power', None) is not None:
                power_true = torch.from_numpy(dataset.target_power).float().to(self.device)
                power_pred_np = power_pred.detach().cpu().numpy()
                np.save(file='result/power_pred', arr=power_pred_np)
                np.save(file='result/power_true', arr=dataset.target_power)
                loss_power = self.power_loss_fn(power_pred, power_true).item()

        return loss_u, loss_v, loss_power

class dataset_package(Dataset):
    def __init__(self, train_x, train_y, train_power=None):
        super().__init__()
        self.input = train_x
        self.target = train_y
        self.target_power = train_power

    def GetDataShape(self):
        shapes = {'input': self.input.shape,
                  'target': self.target.shape}
        if self.target_power is not None:
            shapes['target_power'] = self.target_power.shape
        return shapes

    def __len__(self, ):
        return self.input.shape[0]

    def __getitem__(self, idx):
        if self.target_power is None:
            return self.input[idx], self.target[idx]
        return self.input[idx], self.target[idx], self.target_power[idx]

########################################################################################################################

if __name__ == '__main__':
    print('Configs:\n', configs.__dict__)
    
    uv_test   = np.load("data/Northeast/uv100_test.npy").astype(np.float32)
    zt_test  = np.load("data/Northeast/1000zt_test.npy").astype(np.float32)
    uv_test   = np.concatenate((uv_test, zt_test), axis=1)
    del zt_test
    
    ele = np.load('data/Northeast/DEM_northeast.npy')

    ele[ele < 0] = 0
    ele= (ele - ele.mean()) / ele.std()

    print('processing test set')
    uv_windows = data_process(uv_test, samples_gap=6)
    del uv_test

    indices = uv_windows.indices
    test_x = uv_windows[:, :24, :, :, :]
    test_y = uv_windows[:, 24:, :, :, :]

    power_test = None
    power_path = configs.power_test_data_path or configs.power_data_path
    if power_path is not None:
        if not os.path.exists(power_path):
            raise FileNotFoundError(f'Power data file not found: {power_path}')
        raw_power = np.load(power_path).astype(np.float32)
        power_sequences = build_power_sequences(raw_power, indices)
        power_test = power_sequences[:, 24:, :][..., None]
        del raw_power, power_sequences

    dataset_test = dataset_package(train_x=test_x, train_y=test_y, train_power=power_test)
    del test_x, test_y, power_test, uv_windows
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())

    trainer = Trainer(configs)
    net = torch.load('chkfile/checkpoint_mfwpn.chk')
    trainer.network.load_state_dict(net['net'])
    
    elev = torch.tensor(ele)
    data = DataLoader(dataset_test, batch_size=1, shuffle=False)
    loss_u_test_0, loss_v_test_0, loss_power_test_0 = trainer.infer(dataset=dataset_test, dataloader=data, ele=elev)

    loss_test_0 = loss_u_test_0 + loss_v_test_0
    if loss_power_test_0 is not None:
        loss_test_0 += getattr(configs, 'power_loss_weight', 1.0) * loss_power_test_0
    msg = 'test loss: {:.4f}, {:.4f}'.format(loss_u_test_0, loss_v_test_0)
    if loss_power_test_0 is not None:
        msg += ', power: {:.4f}'.format(loss_power_test_0)
    msg += ', total: {:.4f}'.format(loss_test_0)
    print(msg)
