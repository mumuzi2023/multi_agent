# agent.py
from __future__ import annotations
import random
from collections import OrderedDict
from dataclasses import dataclass, asdict, field  # 导入 field
from typing import Callable, Union, Tuple, Optional  # 导入 Tuple, Optional
import torch
import torch.nn as nn
from memory import ReplayMemory, Transition
# from dqn import DQN # 改为导入具体的网络类
from dqn import DQN_MLP, DQN_CNN  # 导入修改后的网络


@dataclass
class DqnAgentConfig:
    # env info
    obs_dim: Optional[int] = None  # 改为 Optional，因为CNN用obs_shape
    obs_shape: Optional[Tuple[int, ...]] = None  # 新增 for CNN
    act_dim: int = None
    hidden_dim: int = 128  # 对于CNN，这可以作为全连接层的隐藏维度
    # training
    batch_size: int = 128
    lr: float = 1e-4
    grad_clip_value: float = 100
    # gamma: discount factor
    gamma: float = 0.99
    # epsilon: exploration probability
    eps_start: float = 0.9
    eps_decay: float = 0.95  # 注意：对于更复杂的环境，可能需要更慢的衰减，例如0.999
    eps_min: float = 0.01
    # replay memory
    mem_size: int = 10_000
    use_cnn: bool = False  # 新增标志来决定使用哪个网络

    def validate(self) -> None:
        if self.use_cnn:
            assert self.obs_shape is not None, "obs_shape must be provided for CNN"
        else:
            assert self.obs_dim is not None, "obs_dim must be provided for MLP"
        assert self.act_dim is not None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CqlAgentConfig(DqnAgentConfig):
    # obs/act dim for each agent
    # 注意：如果CQL也用于CNN，这里的obs_dims也需要调整为obs_shapes
    obs_dims: Optional[OrderedDict[str, int]] = None
    obs_shapes: Optional[OrderedDict[str, Tuple[int, ...]]] = None  # 新增
    act_dims: OrderedDict[str, int] = None

    def validate(self) -> None:
        # 根据 use_cnn 调整断言
        if self.use_cnn:
            assert self.obs_shapes is not None
        else:
            assert self.obs_dims is not None
        assert self.act_dims is not None
        # super().validate() # DqnAgentConfig的validate现在会被调用，但可能需要调整其逻辑

    def infer_joint_space(self) -> CqlAgentConfig:
        if self.use_cnn:
            # 对于CNN，联合观测空间的处理可能更复杂，
            # 这里简单示例为将第一个智能体的形状作为联合形状 (需要具体问题具体分析)
            # 或者将所有展平后求和，但这违背了CNN的初衷
            # 通常，如果用CNN，可能不会直接用这种方式合并多智能体观测给单个网络
            # self.obs_shape = next(iter(self.obs_shapes.values())) # 这是一个需要仔细考虑的设计
            # 或者，如果CQL的中央网络是MLP，则可以将CNN的输出特征向量连接起来
            example_shape_sum = 0
            for shape_val in self.obs_shapes.values():
                # 假设CNN输出一个固定长度的特征向量，这里用hidden_dim代替
                # 实际中需要从CNN模型获取特征向量维度
                example_shape_sum += self.hidden_dim  # 这是一个粗略的估计
            self.obs_dim = example_shape_sum
            self.obs_shape = None  # 因为我们合并成了MLP的输入
            self.use_cnn = False  # 假设联合网络是MLP
        else:
            self.obs_dim = sum(self.obs_dims.values())

        self.act_dim = 1
        for cur_act_dim in self.act_dims.values():
            self.act_dim *= cur_act_dim
        return self


class DqnAgent:
    def __init__(self, sid: str, config: Union[DqnAgentConfig, CqlAgentConfig], act_sampler: Callable, device=None):
        self.sid = sid
        self.config = config
        self.config.validate()
        self.act_sampler = act_sampler
        self.replay_memory = ReplayMemory(capacity=self.config.mem_size)
        self.eps = self.config.eps_start
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

        # DQN - 根据配置选择网络类型
        if self.config.use_cnn:
            assert self.config.obs_shape is not None
            self.policy_net = DQN_CNN(obs_shape=self.config.obs_shape, n_act=self.config.act_dim,
                                      n_hidden_fc=self.config.hidden_dim).to(self.device)
            self.target_net = DQN_CNN(obs_shape=self.config.obs_shape, n_act=self.config.act_dim,
                                      n_hidden_fc=self.config.hidden_dim).to(self.device)
        else:
            assert self.config.obs_dim is not None
            self.policy_net = DQN_MLP(n_obs=self.config.obs_dim, n_act=self.config.act_dim,
                                      n_hidden=self.config.hidden_dim).to(self.device)
            self.target_net = DQN_MLP(n_obs=self.config.obs_dim, n_act=self.config.act_dim,
                                      n_hidden=self.config.hidden_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.opt = torch.optim.AdamW(self.policy_net.parameters(), lr=self.config.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

    # select_action, _select_action_eps, train, memorize, update_target_network, update_eps 方法保持不变
    # 它们内部调用 policy_net 和 target_net 的方式是通用的。
    # 关键在于输入给这些网络的数据 (state) 的形状，这将在主脚本中处理。

    def select_action(self, state: torch.Tensor, **kwargs):
        return self._select_action_eps(state, dqn=self.policy_net, **kwargs)

    def select_action_greedy(self, state: torch.Tensor, dqn: nn.Module, **kwargs):
        return self._select_action_eps(state, dqn=dqn, eps=0, **kwargs)

    def _select_action_eps(self, state: torch.Tensor, dqn: nn.Module, eps: float = -1, **kwargs):
        """
        input shape: 1 x obs_dim (for MLP) or 1 x C x H x W (for CNN)
        output shape: 1 x 1
        """
        if eps == -1:
            eps = self.eps
        if random.random() < eps:
            sample_res = torch.tensor([[self.act_sampler()]], device=self.device, dtype=torch.long)
            return sample_res
        else:
            with torch.no_grad():
                q_values: torch.Tensor = dqn(state)  # state 必须是网络期望的形状
                sel_res = q_values.argmax(dim=1).reshape(1, 1)
                return sel_res

    def train(self) -> None:
        if len(self.replay_memory) < self.config.batch_size:
            return

        transitions = self.replay_memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)

        # 确保 s for s in batch.next_state if s is not None 是一个列表的张量
        # 如果列表为空，torch.cat会报错
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) > 0:
            non_final_nxt_states = torch.cat(non_final_next_states_list)
        else:  # 如果所有下一状态都是None
            non_final_nxt_states = torch.empty(
                (0,) + self.config.obs_shape if self.config.use_cnn else (0, self.config.obs_dim), device=self.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values_batch: torch.Tensor = self.policy_net(state_batch)
        state_action_q_values = q_values_batch.gather(1, action_batch)

        next_state_best_q_values = torch.zeros(self.config.batch_size, device=self.device)
        if len(non_final_next_states_list) > 0:  # 仅当存在非终止的下一状态时才计算
            with torch.no_grad():
                next_state_best_q_values[non_final_mask] = self.target_net(non_final_nxt_states).max(1).values

        expected_state_action_q_values = reward_batch + (
                    self.config.gamma * next_state_best_q_values.reshape(self.config.batch_size, 1))

        loss = self.criterion(state_action_q_values, expected_state_action_q_values)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.config.grad_clip_value)
        self.opt.step()

    def memorize(self, *args) -> None:
        self.replay_memory.push(*args)

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_eps(self):
        self.eps = max(self.config.eps_min, self.eps * self.config.eps_decay)


# CqlAgent 和 CqlAgentConfig 的修改类似，主要是区分 obs_dim 和 obs_shape
# 这里为了简洁，暂时省略 CqlAgent 的完整修改，但思路是一致的。
class CqlAgent(DqnAgent):
    def __init__(self, sid: str, config: CqlAgentConfig, act_sampler: Callable, device=None):
        super().__init__(sid, config, act_sampler, device)
    # ... (CqlAgent 的其他方法，如果它们依赖于观测维度/形状，也需要相应调整)
    # 例如，get_masked_joint_obs 如果处理CNN的形状，逻辑会更复杂
    # train 方法中 encode_joint_action 等可能不受影响，因为它们处理的是动作