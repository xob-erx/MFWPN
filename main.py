import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from openstl.models.mfwpn import MFWPN_Model
from config import configs
import pickle
import math
from utils.data_sliding import *
from utils import SSIM
from sklearn.model_selection import train_test_split

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
        self.power_loss_weight = getattr(configs, 'power_loss_weight', 1.0)

    def loss(self, y_pred, y_true, idx):
        if idx == 'u':
            idx = 0
        if idx == 'v':
            idx = 1
            
        rmse = torch.mean((y_pred[:, :, idx] - y_true[:, :, idx])**2, dim=[2, 3])
        rmse = torch.mean(torch.sqrt(rmse.mean(dim=0)))
            
        return rmse
    
    def SSIM_loss(self, pred, true):
        pred_np = pred[:,:,:2].permute(1,0,2,3,4)
        true_np = true[:,:,:2].permute(1,0,2,3,4)
        total_loss = 0.0
 
        for i in range(pred_np.shape[0]):
                  loss = 1 - SSIM.SSIM(pred_np[i], true_np[i])
                  total_loss += loss.item()

        average_loss = total_loss / (pred_np.shape[0])
        return average_loss

    def Angle_loss(self, batch_y, pred_y):
        true = self.Angle_wind(batch_y)
        pred = self.Angle_wind(pred_y)
        
        diff = torch.abs(true - pred)
        diff = torch.where(diff > 180, 360 - diff, diff)

        diff_normalized = diff / 180.0

        mse = torch.mean(diff_normalized ** 2)
        rmse = torch.sqrt(mse)
        return rmse
    
    def Angle_wind(self, batch_y):
        true = torch.tensor(batch_y, dtype=torch.float)
        a_fushu = true[:, :, 0, :, :]
        b_fushu = true[:, :, 1, :, :]
        complex_tensor = a_fushu + 1j * b_fushu
        angle_rad = torch.angle(complex_tensor)
        angle_deg = angle_rad * (180 / 3.141592653589793)
        angle_metric = angle_deg.unsqueeze(2)
        return angle_metric
    
    def train_once(self, input_uv, uv_true, power_true, ssr_ratio, ele):
        outputs = self.network(input_uv.float().to(self.device), ele.float().to(self.device))
        uv_pred = outputs['wind']
        power_pred = outputs.get('power')
        self.opt.optimizer.zero_grad()
        loss_u = self.loss(uv_pred, uv_true.float().to(self.device), self.u)
        loss_v = self.loss(uv_pred, uv_true.float().to(self.device), self.v)

        loss_ssim_value = self.SSIM_loss(uv_pred, uv_true.float().to(self.device))
        loss_ssim = torch.tensor(loss_ssim_value, device=self.device)
        loss_angle = self.Angle_loss(uv_true.float().to(self.device), uv_pred)
        loss_angle_value = loss_angle.item()

        total_loss = loss_u + loss_v + loss_ssim + loss_angle
        power_loss_value = None
        if power_pred is not None and power_true is not None:
            power_target = power_true.float().to(self.device)
            power_loss = self.power_loss_fn(power_pred, power_target)
            total_loss = total_loss + self.power_loss_weight * power_loss
            power_loss_value = power_loss.item()

        total_loss.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.opt.step()

        return (loss_u.item(), loss_v.item(), loss_ssim_value, loss_angle_value,
                power_loss_value, total_loss.item())

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

            loss_u = self.loss(uv_pred, uv_true, self.u).item()
            loss_v = self.loss(uv_pred, uv_true, self.v).item()

            loss_power = None
            if power_pred is not None and getattr(dataset, 'target_power', None) is not None:
                power_true = torch.from_numpy(dataset.target_power).float().to(self.device)
                loss_power = self.power_loss_fn(power_pred, power_true).item()

        return loss_u, loss_v, loss_power

    def train(self, dataset_train, dataset_eval, elev, chk_path):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)
        elev = torch.tensor(elev)
          
        count = 0
        best = 100
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            # train
            self.network.train()
            for j, batch in enumerate(dataloader_train):
                if len(batch) == 2:
                    input_uv, uv_true = batch
                    power_true = None
                else:
                    input_uv, uv_true, power_true = batch

                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)

                loss_u, loss_v, loss_ssim, loss_angle, loss_power, loss_total = \
                    self.train_once(input_uv, uv_true, power_true, ssr_ratio, elev)

                if (j+1) % self.configs.display_interval == 0:
                    msg = 'batch training loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                        loss_u, loss_v, loss_ssim, loss_angle)
                    if loss_power is not None:
                        msg += ', power: {:.4f}'.format(loss_power)
                    msg += ', total: {:.4f}, ssr: {:.5f}, lr: {:.5f}'.format(
                        loss_total, ssr_ratio, self.opt.rate())
                    print(msg)

                if (i+1 >= 10) and (j+1)%(self.configs.display_interval * 2) == 0:
                    loss_u_eval_0, loss_v_eval_0, loss_power_eval_0 = self.infer(
                        dataset=dataset_eval, dataloader=dataloader_eval, ele=elev)
                    loss_eval_0 = loss_u_eval_0 + loss_v_eval_0
                    if loss_power_eval_0 is not None:
                        loss_eval_0 += self.power_loss_weight * loss_power_eval_0
                    msg = 'batch eval loss: {:.4f}, {:.4f}'.format(loss_u_eval_0, loss_v_eval_0)
                    if loss_power_eval_0 is not None:
                        msg += ', power: {:.4f}'.format(loss_power_eval_0)
                    msg += ', total: {:.4f}'.format(loss_eval_0)
                    print(msg)

                    if loss_eval_0 < best:
                        self.save_model(chk_path)
                        best = loss_eval_0
                        count = 0
                        print('saving model')


            loss_u_eval, loss_v_eval, loss_power_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval, ele=elev)
            loss_eval = loss_u_eval + loss_v_eval
            if loss_power_eval is not None:
                loss_eval += self.power_loss_weight * loss_power_eval
            msg = 'epoch eval loss: {:.4f}, {:.4f}'.format(loss_u_eval, loss_v_eval)
            if loss_power_eval is not None:
                msg += ', power: {:.4f}'.format(loss_power_eval)
            msg += ', total: {:.4f}'.format(loss_eval)
            print(msg)


            if loss_eval >= best:
                count += 1
                print('eval loss is not reduced for {} epoch'.format(count))
                print('best is {} until now'.format(best))
            else:
                count = 0
                print('eval loss is reduced from {:.5f} to {:.5f}, saving model'.format(best, loss_eval))
                self.save_model(chk_path)
                best = loss_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.opt.optimizer.state_dict()}, path)

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

    def split_data(self, test_size=0.1, random_state=35):
        """
        Split the data into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
            random_state (int or None): Random seed for reproducibility (default is None).

        Returns:
            (train_dataset, test_dataset): Tuple of Dataset objects for training and testing.
        """
        # Use sklearn's train_test_split to randomly split the data
        if self.target_power is not None:
            X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
                self.input, self.target, self.target_power,
                test_size=test_size, random_state=random_state
            )

            train_dataset = dataset_package(X_train, y_train, p_train)
            test_dataset = dataset_package(X_test, y_test, p_test)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.input, self.target, test_size=test_size, random_state=random_state
            )

            train_dataset = dataset_package(X_train, y_train)
            test_dataset = dataset_package(X_test, y_test)

        return train_dataset, test_dataset
########################################################################################################################

if __name__ == '__main__':
    print('Configs:\n', configs.__dict__)

    uv_train_path = configs.stage1_uv_train_path
    zt_train_path = configs.stage1_zt_train_path
    dem_path = configs.stage1_dem_path
    checkpoint_path = configs.stage1_checkpoint_path
    config_dump_path = configs.stage1_train_config_dump_path
    samples_gap = configs.stage1_samples_gap

    for data_path in [uv_train_path, zt_train_path, dem_path]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Stage1 data file not found: {data_path}')

    uv_train = np.load(uv_train_path).astype(np.float32)
    zt_train = np.load(zt_train_path).astype(np.float32)
    
    uv_train = np.concatenate((uv_train, zt_train), axis=1)
    del zt_train
    
    ele = np.load(dem_path).astype(np.float32)

    ele[ele < 0] = 0
    ele= (ele - ele.mean()) / ele.std()

    print(f'Loading stage1 data: uv={uv_train_path}, zt={zt_train_path}, dem={dem_path}')
    print('processing training set')
    uv_windows = data_process(uv_train, samples_gap=samples_gap)
    del uv_train

    indices = uv_windows.indices
    train_x = uv_windows[:, :24, :, :, :]
    train_y = uv_windows[:, 24:, :, :, :]

    power_train = None
    if configs.power_data_path is not None:
        power_path = configs.power_data_path
        if not os.path.exists(power_path):
            raise FileNotFoundError(f'Power data file not found: {power_path}')
        raw_power = np.load(power_path).astype(np.float32)
        power_sequences = build_power_sequences(raw_power, indices)
        power_train = power_sequences[:, 24:, :][..., None]
        del raw_power, power_sequences
    elif configs.turbine_coords:
        print('Warning: turbine coordinates provided but no power data path set; power supervision will be skipped.')

    dataset_full = dataset_package(train_x=train_x, train_y=train_y, train_power=power_train)
    del train_x, train_y, power_train

    del uv_windows

    dataset_train, dataset_val = dataset_full.split_data()

    print('Dataset_train Shape:\n', dataset_train.GetDataShape())
    print('Dataset_val Shape:\n', dataset_val.GetDataShape())
    
    trainer = Trainer(configs)
    trainer.save_configs(config_dump_path)
    
    trainer.train(dataset_train, dataset_val, ele, checkpoint_path)
