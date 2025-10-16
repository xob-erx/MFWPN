import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size_test = 4
configs.batch_size = 4
configs.weight_decay = 0
configs.display_interval = 100
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 10
configs.gradient_clipping = False
configs.clipping_threshold = 1.
configs.warmup = 5000
configs.d_model = 256
configs.input_gap = 1
configs.pred_shift = 24
configs.dropout = 0.2
configs.ssr_decay_rate = 5.e-5
configs.turbine_coords = None  # e.g., [(h0, w0), (h1, w1), ...]
configs.power_roi_size = 5
configs.power_conv_channels = 32
configs.power_mlp_hidden = 64
configs.power_loss_weight = 1.0
configs.power_data_path = None
configs.power_test_data_path = None
configs.feature_hw = (64, 80)
