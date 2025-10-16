import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import numbers
from typing import Iterable, Optional, Sequence, Tuple
from einops import rearrange
from openstl.modules import (ConvSC, GASubBlock)


def _validate_turbine_coords(coords: Iterable[Tuple[int, int]], height: int, width: int) -> Sequence[Tuple[int, int]]:
    validated = []
    for idx, coord in enumerate(coords):
        if len(coord) != 2:
            raise ValueError(f"Turbine coordinate at index {idx} must contain two values (h, w), got {coord}.")
        h, w = int(coord[0]), int(coord[1])
        if not (0 <= h < height and 0 <= w < width):
            raise ValueError(
                f"Turbine coordinate {(h, w)} is outside the feature map with shape ({height}, {width}).")
        validated.append((h, w))
    return validated


class TurbinePowerHead(nn.Module):
    """Predict turbine power from shared spatiotemporal features."""

    def __init__(self, in_channels: int, turbine_coords: Sequence[Tuple[int, int]], roi_size: int = 5,
                 conv_channels: int = 32, mlp_hidden_dim: int = 64, feature_hw: Tuple[int, int] = (64, 80)):
        super().__init__()
        if roi_size % 2 == 0 or roi_size <= 0:
            raise ValueError(f"roi_size must be a positive odd number, got {roi_size}.")

        self.roi_size = roi_size
        self.pad = roi_size // 2
        self.feature_hw = feature_hw
        self.turbine_coords = _validate_turbine_coords(turbine_coords, *feature_hw)
        self.num_turbines = len(self.turbine_coords)

        conv_hidden = max(conv_channels, in_channels)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute power predictions.

        Args:
            features: Tensor of shape [B, T, C, H, W].

        Returns:
            Power predictions shaped [B, T, N_turbines, 1].
        """
        b, t, c, h, w = features.shape
        if (h, w) != self.feature_hw:
            raise ValueError(f"Unexpected feature map size {(h, w)}; expected {self.feature_hw} for turbine decoding.")

        x = features.reshape(b * t, c, h, w)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')

        power_outputs = []
        for h_idx, w_idx in self.turbine_coords:
            center_h = h_idx + self.pad
            center_w = w_idx + self.pad
            patch = x[:, :, center_h - self.pad:center_h + self.pad + 1,
                      center_w - self.pad:center_w + self.pad + 1]
            local_feat = self.local_conv(patch)
            pooled = self.pool(local_feat).reshape(b, t, -1)
            regressed = self.regressor(pooled.reshape(b * t, -1)).reshape(b, t, 1)
            power_outputs.append(regressed)

        return torch.stack(power_outputs, dim=2)


class MFWPN_Model(nn.Module):
    def __init__(self, hid_S=2, hid_T=256, N_S=2, N_T=8, model_type='gsta',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, turbine_coords: Optional[Sequence[Tuple[int, int]]] = None,
                 power_roi_size: int = 5, power_conv_channels: int = 32, power_mlp_hidden: int = 64,
                 feature_hw: Tuple[int, int] = (64, 80), **kwargs):
        super(MFWPN_Model, self).__init__()
        T, C, H, W = 24, 2, 64, 80  # T is pre_seq_length
        act_inplace = False
        self.enc = Encoder(spatio_kernel=spatio_kernel_enc, act_inplace=True)
        self.dec = Decoder(spatio_kernel=spatio_kernel_dec, act_inplace=True)
        self.hid_w = MidMetaNet(T*hid_S, hid_T, N_T, input_resolution=(64, 80), model_type=model_type,
                                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.hid_tz = MidMetaNet(T*hid_S, hid_T, N_T, input_resolution=(64, 80), model_type=model_type,
                                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.channel_f = CAM(channel = 48)
        self.channel_i = CAM(channel = 48)
        self.gate = nn.Tanh()
        self.ele_conv = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, 48, kernel_size=1, stride=1))
        self.power_head: Optional[TurbinePowerHead] = None
        if turbine_coords:
            self.power_head = TurbinePowerHead(
                in_channels=hid_S,
                turbine_coords=turbine_coords,
                roi_size=power_roi_size,
                conv_channels=power_conv_channels,
                mlp_hidden_dim=power_mlp_hidden,
                feature_hw=feature_hw
            )

    def forward(self, x_raw, ele, **kwargs):
        B, T, C, H, W = x_raw.shape

        ele = ele[None, None, :, :]
        ele_h = ele
        ele = ele.repeat(B*T,1,1,1)
        x = x_raw.view(B*T, C, H, W)  
        wcs, tzcs = self.enc(x, ele)
        
        _, C_w, H_, W_ = wcs.shape
        _, C_tz, H_, W_ = tzcs.shape

        w = wcs.view(B, T, C_w, H_, W_)
        tz = tzcs.view(B, T, C_tz, H_, W_)

        hid_w = self.hid_w(w)  
        hid_tz = self.hid_tz(tz)
        #ele_h = ele_h.repeat(B, 48, 1, 1)
        ####time fusion
        hid_w = hid_w * self.channel_f(hid_tz) + self.channel_i(hid_tz) * self.gate(hid_tz)  
        hid_w = hid_w.reshape(B , T, C_w, H_, W_)

        power_pred = None
        if self.power_head is not None:
            power_pred = self.power_head(hid_w)

        dec_input = hid_w.reshape(B * T, C_w, H_, W_)
        Y = self.dec(dec_input)
        Y = Y.reshape(B, T, 2, H, W)

        outputs = {'wind': Y}
        if power_pred is not None:
            outputs['power'] = power_pred
        return outputs
    
class SAM(nn.Module):
    def __init__(self, spatial_kernel=7):
        super(SAM, self).__init__()
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)      
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return spatial_out

class CAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAM, self).__init__()
   
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(       
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),   
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )        
        self.sigmoid = nn.Sigmoid()
                                   
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        eca = max_out + avg_out
        channel_out = self.sigmoid(eca)                           
        return channel_out
                                  
def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    """3D Encoder for SimVP"""
    def __init__(self, spatio_kernel, act_inplace=True):
        super(Encoder, self).__init__()  
        
        self.wscale = nn.Parameter(torch.ones(2))
        self.tzscale = nn.Parameter(torch.ones(2))
         ####Conv branch
        self.wce =  INN_all(num_layers=1)
        self.tzce = INN_all(num_layers=1)
        ####Self-attention branch
        self.wse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')
        
        self.tzse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')   
        self.sa_f = SAM()
        self.sa_i = SAM()
        self.gate_1 = nn.Sigmoid()
        self.gate_2 = nn.Tanh()
        ####ele
        self.ele_conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(2, 2, kernel_size=1, stride=1))
        
    def forward(self, x, ele):
        w = x[:, 0:2]
        tz = x[:, 2:4]
        
        wce_x = self.wce(w)
        wse_x = self.wse(w)
        wcs = wse_x * self.wscale[0] + wce_x * self.wscale[1]

        tzce_x = self.tzce(tz)
        tzse_x = self.tzse(tz)
        tzcs = tzse_x * self.tzscale[0] + tzce_x * self.tzscale[1]
        ####Fusion module
        tz_gate_2 = self.gate_2(tzcs)
        wcs = wcs * self.sa_f(tzcs) + self.sa_i(tzcs) * self.gate_2(tzcs) + wcs * self.gate_1(self.ele_conv(ele))

        return wcs, tzcs

class Decoder(nn.Module):
    """3D Encoder for SimVP"""
    def __init__(self, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        
        self.dwscale = nn.Parameter(torch.ones(2))
        ####Conv branch
        self.dwce =  INN_all(num_layers=1)                   
        ####Self-attention branch
        self.dwse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')
    def forward(self, wdcs):
        
        dwse_x = self.dwse(wdcs)
        dwce_x = self.dwce(wdcs)
        dwcs = dwse_x * self.dwscale[0] + dwce_x * self.dwscale[1]
        
        return dwcs

class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'
        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"
            
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)
    
class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]
        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1)) 
        self.enc_1 = nn.Sequential(*enc_layers)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc_1[i](z)
        return z



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out 


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
    
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
        
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class INN(nn.Module):
    def __init__(self, input, output, ratio):
        super(INN, self).__init__()
        hidden_dim = int(input * ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(input, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, output, 1, bias=False),
            nn.BatchNorm2d(output),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.phi = INN(input=32, output=32, ratio=2)
        self.seta = INN(input=32, output=32, ratio=2)
        
    def forward(self, f1, f2):
        f2 = f2 + self.phi(f1)
        f1 = f1 + self.seta(f2)
        return f1, f2

class INN_all(nn.Module):
    def __init__(self, num_layers=None):
        super(INN_all, self).__init__()
        INN_layers = [Feature() for _ in range(num_layers)]
        self.net = nn.Sequential(*INN_layers)
        
        self.shffle = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        
    def separate(self, x):
        f1, f2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return f1, f2
    
    def forward(self, x):
        f1, f2 = self.separate(self.shffle(x))
        
        for layer in self.net: 
            f1, f2 = layer(f1, f2)
        f_out = self.fusion(torch.cat((f1, f2), dim=1))
        return f_out

