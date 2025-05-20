import torch
import torch.nn as nn

def _init_weights(layer: nn.Module, nonlinearity: str) -> None:
  if isinstance(layer, nn.Linear):
    nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
    if layer.bias is not None:
      nn.init.zeros_(layer.bias)

class DQN(nn.Module):
  def __init__(self, n_obs: int, n_act: int, n_hidden: int = 128):
    super(DQN, self).__init__()
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
    # re-initialize learnable parameters
    for layer in self.model:
      _init_weights(layer, nonlinearity='relu')

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    res = self.model(x)
    return res
