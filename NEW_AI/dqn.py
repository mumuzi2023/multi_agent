# dqn.py
import torch
import torch.nn as nn
import numpy as np  # 确保导入numpy


def _init_weights(layer: nn.Module, nonlinearity: str) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):  # 添加nn.Conv2d
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class DQN_MLP(nn.Module):  # 将原来的DQN重命名为DQN_MLP
    def __init__(self, n_obs: int, n_act: int, n_hidden: int = 128):
        super(DQN_MLP, self).__init__()
        self.obs_dim = n_obs
        self.act_dim = n_act
        self.hidden_dim = n_hidden
        # model
        self.model = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.act_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in self.model:
            _init_weights(layer, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入是图像格式 NCHW 且被意外展平了，这里要确保它是正确的 MLP 输入
        if x.ndim > 2:
            x = x.reshape(x.size(0), -1)  # 确保展平
        res = self.model(x)
        return res


class DQN_CNN(nn.Module):
    def __init__(self, obs_shape: tuple, n_act: int, n_hidden_fc: int = 128):
        super(DQN_CNN, self).__init__()
        self.obs_shape = obs_shape  # e.g., (C, H, W)
        self.n_act = n_act
        self.n_hidden_fc = n_hidden_fc

        # 假设 obs_shape 是 (C, H, W)
        c, h, w = obs_shape

        # 卷积层示例 - 你需要根据实际的 H, W 调整这些参数
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),  # (H, W)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (H/4, W/4)
        )

        # 计算卷积层输出后的扁平化尺寸
        # 需要根据你的obs_shape和卷积/池化层参数手动计算或通过一次前向传播获取
        # 这里我们用一个虚拟输入来计算
        dummy_input = torch.zeros(1, c, h, w)
        conv_out_dim = self._get_conv_out_dim(dummy_input)

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_dim, n_hidden_fc),
            nn.ReLU(),
            nn.Linear(n_hidden_fc, n_act)
        )
        self._reset_parameters()

    def _get_conv_out_dim(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            x = self.conv_layers(x)
        return int(np.prod(x.shape[1:]))  # C_out * H_out * W_out

    def _reset_parameters(self) -> None:
        for block in [self.conv_layers, self.fc_layers]:
            for layer in block:
                _init_weights(layer, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入 x 的形状是 (N, C, H, W)
        if x.ndim == 3:  # 如果是 (C,H,W)
            x = x.unsqueeze(0)
        elif x.ndim == 2 and np.prod(x.shape[1:]) == np.prod(self.obs_shape):  # 如果是 (N, C*H*W)
            x = x.reshape(x.size(0), *self.obs_shape)

        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)  # 扁平化
        res = self.fc_layers(x)
        return res

# 保留原来的DQN名称，但让它根据obs_dim的类型决定使用MLP还是CNN
# 或者，更清晰的方式是在agent.py中决定加载哪个网络
# 这里我们保持分离，DQN_MLP 和 DQN_CNN