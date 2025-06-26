from typing import Any, Dict, List, Optional, Type

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs, self.features_extractor))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def reset_parameters(self):
        """
        Reset the network parameters randomly.
        """
        self.q_net = nn.Sequential(
            *create_mlp(self.features_dim, int(self.action_space.n), self.net_arch, self.activation_fn)
        )


class RNNQNetwork(QNetwork):
    """
    Same as QNetwork but with a one‐step RNN prior to the MLP head.
    All other methods (_predict, _get_constructor_parameters, reset_parameters)
    are inherited directly from QNetwork.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
    ) -> None:
        # initialize features_extractor + MLP defaults
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers  = rnn_num_layers

        # override the pure-MLP q_net with an RNN → MLP
        self.rnn = nn.RNN(
            input_size=self.features_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        # post-RNN MLP head to actions
        layers = create_mlp(
            input_dim=rnn_hidden_size,
            output_dim=self.action_space.n,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )
        self.post_rnn = nn.Sequential(*layers)

        # placeholder for hidden state; will be (num_layers, batch, hidden_size)
        self._h = None

    def reset_hidden(self, batch_size: int = 1, device: th.device = None) -> None:
        """Zero out the hidden state. Call this at the start of each new episode."""
        device = device or next(self.parameters()).device
        self._h = th.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # 1) extract features
        features = self.extract_features(obs, self.features_extractor)
        # 2) add time-dim: (batch, seq=1, feat_dim)
        rnn_in = features.unsqueeze(1)

        # 3) if this is the very first call (or after reset), init hidden
        if self._h is None or self._h.size(1) != obs.size(0):
            # assume batch = obs.shape[0]
            self.reset_hidden(batch_size=obs.size(0), device=obs.device)

        # 4) run RNN with the *previous* hidden state
        #    out: (batch, seq=1, hidden), h_n: (num_layers, batch, hidden)
        out, h_n = self.rnn(rnn_in, self._h)

        # 5) detach and cache the new hidden state for next call
        self._h = h_n.detach()

        # 6) squash seq dim and feed through your MLP head
        out = out.squeeze(1)            # (batch, hidden)
        return self.post_rnn(out)       # (batch, num_actions)
    

class EPLHbNetwork(QNetwork):
    """
    Same as QNetwork but with a one‐step RNN prior to the MLP head and an EPLHb layer.
    All other methods (_predict, _get_constructor_parameters, reset_parameters)
    are inherited directly from QNetwork.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        eplhb_hidden_dim: int = 32,
    ) -> None:
        # initialize features_extractor + MLP defaults
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers  = rnn_num_layers
        self.eplhb_hidden_dim = eplhb_hidden_dim
        self.eplhb_coeff = nn.Parameter(th.tensor(0.1))  # start at 0.1, for instance

        # override the pure-MLP q_net with an RNN → MLP
        self.rnn = nn.RNN(
            input_size=self.features_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        # post-RNN MLP head to actions
        layers = create_mlp(
            input_dim=rnn_hidden_size,
            output_dim=self.action_space.n,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )
        self.post_rnn = nn.Sequential(*layers)

        # --- NEW: EPLHb MLP ---
        # input is [rnn_hidden + Q-MLP pre-output], map to a scalar
        self.eplhb = nn.Sequential(
            nn.Linear(rnn_hidden_size + self.action_space.n, self.eplhb_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.eplhb_hidden_dim, 1)
        )

        # placeholder for hidden state; will be (num_layers, batch, hidden_size)
        self._h = None

    def reset_hidden(self, batch_size: int = 1, device: th.device = None) -> None:
        """Zero out the hidden state. Call this at the start of each new episode."""
        device = device or next(self.parameters()).device
        self._h = th.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # 1) extract features
        features = self.extract_features(obs, self.features_extractor)
        # 2) add time-dim: (batch, seq=1, feat_dim)
        rnn_in = features.unsqueeze(1)

        # 3) if this is the very first call (or after reset), init hidden
        if self._h is None or self._h.size(1) != obs.size(0):
            # assume batch = obs.shape[0]
            self.reset_hidden(batch_size=obs.size(0), device=obs.device)

        # 4) run RNN with the *previous* hidden state
        #    out: (batch, seq=1, hidden), h_n: (num_layers, batch, hidden)
        out, h_n = self.rnn(rnn_in, self._h)

        # 5) detach and cache the new hidden state for next call
        self._h = h_n.detach()

        # 6) squash seq dim and feed through your MLP head
        out = out.squeeze(1)            # (batch, hidden)
        q_out = self.post_rnn(out)       # (batch, num_actions)

        return q_out

    def forward_full(self, obs: th.Tensor):
        # 1) extract features
        features = self.extract_features(obs, self.features_extractor)
        # 2) add time-dim: (batch, seq=1, feat_dim)
        rnn_in = features.unsqueeze(1)

        # 3) if this is the very first call (or after reset), init hidden
        if self._h is None or self._h.size(1) != obs.size(0):
            # assume batch = obs.shape[0]
            self.reset_hidden(batch_size=obs.size(0), device=obs.device)

        # 4) run RNN with the *previous* hidden state
        #    out: (batch, seq=1, hidden), h_n: (num_layers, batch, hidden)
        out, h_n = self.rnn(rnn_in, self._h)
        last_embed = out[:, -1, :]  # (batch, hidden) - last time step output

        # 5) detach and cache the new hidden state for next call
        self._h = h_n.detach()

        # 6) squash seq dim and feed through your MLP head (why not last embed as input?)
        out = out.squeeze(1)            # (batch, hidden)
        q_out = self.post_rnn(out)       # (batch, num_actions)

        # 7) concat last embed and q_out to feed to eplhb
        concat = th.cat([last_embed, q_out], dim=-1)
        eplhb_out = self.eplhb(concat).squeeze(-1)

        return q_out, h_n, eplhb_out











class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        with_RNN_layer: bool = True,
        with_EPLHb_layer: bool = False,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.with_RNN_layer = with_RNN_layer
        self.with_EPLHb_layer = with_EPLHb_layer

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
 
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)
        self.d_net = self.make_q_net()
        self.d_net.load_state_dict(self.q_net.state_dict())
        self.d_net.set_training_mode(False)  # Keep false so that we don't dropout anything here

        # Set up parameter groups
        main_lr = lr_schedule(1)
        # Allow user to specify eplhb_lr and coeff_lr via optimizer_kwargs
        eplhb_lr = self.optimizer_kwargs.pop('eplhb_lr', 1e-4)
        coeff_lr = self.optimizer_kwargs.pop('coeff_lr', 1e-5)

        from .DQN_policy import EPLHbNetwork
        if isinstance(self.q_net, EPLHbNetwork):
            eplhb_params = list(self.q_net.eplhb.parameters())
            eplhb_coeff_param = [self.q_net.eplhb_coeff]
        else:
            eplhb_params = []
            eplhb_coeff_param = []
        other_params = [
            p for n, p in self.q_net.named_parameters()
            if not n.startswith('eplhb.') and n != 'eplhb_coeff'
        ]

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            [
                {'params': other_params, 'lr': main_lr},
                {'params': eplhb_params, 'lr': eplhb_lr},
                {'params': eplhb_coeff_param, 'lr': coeff_lr},
            ],
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        if self.with_EPLHb_layer: net_cls = EPLHbNetwork
        elif self.with_RNN_layer: net_cls = RNNQNetwork
        else: net_cls = QNetwork
        return net_cls(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        #self.d_net.set_training_mode(mode)
        self.training = mode

    def jump_start_cuda(self) -> None:
        """Surprisingly, this fixes a CUDNN_STATUS_NOT_INITIALIZED error when using DQN with CUDA on Arcade.
        https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch"""
        s = 32
        dev = th.device('cuda')
        th.nn.functional.conv2d(th.zeros(s, s, s, s, device=dev), th.zeros(s, s, s, s, device=dev))

MlpPolicy = DQNPolicy


class CnnPolicy(DQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputPolicy(DQNPolicy):
    """
    Policy class for DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
