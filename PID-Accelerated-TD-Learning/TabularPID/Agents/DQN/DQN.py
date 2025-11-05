import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from copy import deepcopy

import globals
import numpy as np
import torch as th
from gymnasium.wrappers import RecordVideo
from gymnasium import spaces
from torch.nn import functional as F
from torch import nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from TabularPID.Agents.DQN.DQN_policy import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork, EPLHbNetwork
from TabularPID.OptimalRates.EvaluateBuffer import run_simulation

SelfDQN = TypeVar("SelfDQN", bound="PID_DQN")

class EPLHb_DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from the Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self, d_tau, tabular_d,
        gain_adapter,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        dump_buffer: bool = False,
        is_double=False,
        optimal_model=None,
        policy_evaluation=False,
        use_stdp: bool = False,
        stdp_decay: float = 1e-5,
    ) -> None:
        
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.setdefault('with_EPLHb_layer', True)
        if optimizer_kwargs is not None:
            policy_kwargs['optimizer_kwargs'] = optimizer_kwargs

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True
        )
        # The stable baselines wrapped env don't play nice with the RecordVideo Wrapper
        # The simplest solution is to reserve an unwrapped instance for video recording, instead of modifying the API
        # Our additions atop stable baselines:
        self.visualization_env = env
        self.d_tau = d_tau
        self.tabular_d = tabular_d
        self.dump_buffer = dump_buffer
        self.buffer = []  # The buffer we dump, if dump_buffer is True
        self.is_double = is_double
        self.optimal_model = optimal_model
        self.policy_evaluation = policy_evaluation

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()

        # Gain adaptation Code
        self.gain_adapter = gain_adapter
        self.BRs = None
        self.previous_p_update, self.p_update = None, None
        self.previous_i_update, self.i_update = None, None
        self.previous_d_update, self.d_update = None, None
        self.kp, self.ki, self.kd = None, None, None
        
        # STDP training parameters
        self.use_stdp = use_stdp
        self.stdp_decay = stdp_decay
        # Extract eplhb_lr from optimizer_kwargs if available
        if optimizer_kwargs is not None:
            self.eplhb_lr = optimizer_kwargs.get('eplhb_lr', learning_rate if isinstance(learning_rate, float) else learning_rate(1))
        else:
            self.eplhb_lr = learning_rate if isinstance(learning_rate, float) else learning_rate(1)

        self.policy.jump_start_cuda()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target
        self.d_net = self.policy.d_net

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            # Update the gains here once we have started training
            if self._n_calls > self.learning_starts:
                #self.gain_adapter.apply_weight_decay(self.replay_buffer)
                update_size = 50000
                update_size = min(update_size, self.replay_buffer.size())

                replay_data = self.replay_buffer.sample(update_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                
                self.gain_adapter.adapt_gains(replay_data)

            # Update the D network
            polyak_update(self.q_net_target.parameters(), self.d_net.parameters(), self.d_tau)
            # Update the target network
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100, seq_len: int = None, batch_idxs: List[int] = None) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        # If we have an RNN AND seq_len is provided, do truncated BPTT
        is_recurrent = hasattr(self.policy.q_net, "reset_hidden")
        use_bptt = is_recurrent and seq_len is not None
        if use_bptt:
            if batch_idxs is not None:
                # Use BPTT with specific batch indices
                return self._train_recurrent(gradient_steps, batch_idxs=batch_idxs, seq_len=seq_len)
            else:
                # Use BPTT with random sampling
                return self._train_recurrent(gradient_steps, batch_size=batch_size, seq_len=seq_len)
        
        
        losses = []
        l2_lambda = getattr(self, 'l2_lambda', None)
        if l2_lambda is None:
            # Try to get from policy_kwargs or pid_params
            l2_lambda = getattr(self, 'policy_kwargs', {}).get('l2_lambda', 1e-4)

        for _ in range(gradient_steps):
            # Sample replay buffer
            if batch_idxs is not None:
                replay_data = self.replay_buffer.sample(batch_idxs=batch_idxs, env=self._vec_normalize_env)  # type: ignore[union-attr]
                actual_batch_size = len(batch_idxs)
            else:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                actual_batch_size = batch_size

            with th.no_grad():
                next_q_values = self.compute_next_q_values(replay_data, actual_batch_size)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Current Q estimates of the target network
                target_current_q_values = self.q_net_target(replay_data.observations)
                target_current_q_values = th.gather(target_current_q_values, dim=1, index=replay_data.actions.long())

                if self.tabular_d:
                    d_values = replay_data.ds
                else:
                    d_values = self.d_net(replay_data.observations)
                    d_values = th.gather(d_values, dim=1, index=replay_data.actions.long())

                kp, ki, kd, alpha, beta = self.gain_adapter.get_gains(
                    replay_data.observations, replay_data.actions, replay_data
                )
                self.BRs = target_q_values - target_current_q_values
                new_zs = beta * replay_data.zs + alpha * self.BRs

                self.p_update = self.BRs
                self.d_update = target_current_q_values - d_values
                self.i_update = new_zs

                # Update the gains
                self.kp = kp
                self.ki = ki # normally is 0
                self.kd = kd # normally is 0
                target = target_current_q_values + kp * self.p_update + ki * self.i_update + kd * self.d_update

            # Forward pass to get Q and EPLHb heads
            q_pred, _, eplhb_out = self.q_net.forward_full(replay_data.observations)
            # pick Q for the taken actions
            q_taken = th.gather(q_pred, dim=1, index=replay_data.actions.long()).squeeze(-1)

            # pull out your learnable coeff
            # Use eplhb_coeff property directly to keep it in the computation graph
            # Fix shape mismatch: eplhb_out is (batch_size,) but target is (batch_size, 1)
            target = target + self.policy.q_net.eplhb_coeff * eplhb_out.unsqueeze(-1)

            # compute TD-error and base loss
            td_error = q_taken - target.squeeze(-1)
            base_loss = F.smooth_l1_loss(q_taken, target.squeeze(-1))

            # auxiliary loss to see whether EPLHb is similar to d_update value
            d_update_loss = F.smooth_l1_loss(eplhb_out, self.d_update.squeeze(-1))

            # L2 regularization for EPLHb weights
            l2_penalty = 0.0
            if hasattr(self.policy.q_net, 'eplhb') and hasattr(self.policy.q_net.eplhb[0], 'weight'):
                l2_penalty = l2_lambda * th.sum(self.policy.q_net.eplhb[0].weight ** 2)

            # sample a little noise term
            noise = th.randn_like(td_error)

            # final joint loss for main network (without d_update_loss)
            final_loss = (
                base_loss + (noise * td_error).mean() + l2_penalty
            )
            losses.append(final_loss.item())

            # separate loss for eplhb network
            eplhb_loss = d_update_loss

            # Optimize both networks together
            self.policy.optimizer.zero_grad()
            
            # Compute gradients for main network (excluding EPLHb parameters)
            final_loss.backward(retain_graph=True)
            
            # Compute gradients for EPLHb network only
            if hasattr(self.policy.q_net, 'eplhb'):
                # Get EPLHb parameters
                eplhb_params = list(self.policy.q_net.eplhb.parameters())
                if hasattr(self.policy.q_net, 'eplhb_coeff_raw'):
                    eplhb_params.append(self.policy.q_net.eplhb_coeff_raw)
                
                # Compute gradients only for EPLHb parameters
                eplhb_grads = th.autograd.grad(
                    eplhb_loss, eplhb_params, retain_graph=True, create_graph=False, allow_unused=True
                )
                
                # Manually set gradients for EPLHb parameters
                for param, grad in zip(eplhb_params, eplhb_grads):
                    if grad is not None:  # Only update if gradient was computed
                        if param.grad is not None:
                            param.grad += grad
                        else:
                            param.grad = grad
                
                # Apply gradient clipping to EPLHb parameters
                th.nn.utils.clip_grad_norm_(eplhb_params, 1.0)

            # Apply gradient clipping to all parameters
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            # Update all parameters with the optimizer
            self.policy.optimizer.step()
            
            if self.policy_kwargs["fixed_sign"]:
                self.policy.q_net.enforce_signs()
                self.policy.q_net_target.enforce_signs()


        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("rollout/BRs", th.mean(self.BRs).item())
        self.logger.record("rollout/BRs_L2_norm", th.norm(self.BRs).item() / np.sqrt(self.BRs.shape[0]))
        self.logger.dump(step=self.num_timesteps)

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the learning rate for optimizers, but preserve EPLHb learning rate.
        """
        # Log the current learning rate
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        
        for optimizer in optimizers:
            # Get the current learning rate from schedule
            new_lr = self.lr_schedule(self._current_progress_remaining)
            
            # Update learning rates for each parameter group
            for param_group in optimizer.param_groups:
                # Check if this is the EPLHb parameter group (group 1)
                if len(optimizer.param_groups) > 1 and param_group == optimizer.param_groups[1]:
                    # Keep the EPLHb learning rate unchanged
                    continue
                else:
                    # Update the main network learning rate
                    param_group["lr"] = new_lr

    def _capture_activations(self, net, obs_seq, act_seq):
        """
        Capture pre-synaptic and post-synaptic activations for STDP updates.
        Returns a dictionary mapping layer names to (pre_activation, post_activation) tuples.
        """
        activations = {}
        
        # Extract features
        features = net.extract_features(obs_seq, net.features_extractor)
        activations['features'] = (None, features.detach())
        
        # Input projection
        rnn_in = net.input_projection(features)
        activations['input_projection'] = (features.detach(), rnn_in.detach())
        
        # Input norm
        rnn_in_norm = net.input_norm(rnn_in)
        activations['input_norm'] = (rnn_in.detach(), rnn_in_norm.detach())
        
        # RNN - need to handle hidden state
        rnn_in_seq = rnn_in_norm.unsqueeze(1)  # [B, 1, rnn_input_size]
        if net._h is None or net._h.size(1) != obs_seq.size(0):
            net.reset_hidden(batch_size=obs_seq.size(0), device=obs_seq.device)
        
        rnn_out, h_n = net.rnn(rnn_in_seq, net._h)
        net._h = h_n.detach()
        rnn_out_squeezed = rnn_out.squeeze(1)  # [B, rnn_hidden_size]
        activations['rnn'] = (rnn_in_norm.detach(), rnn_out_squeezed.detach())
        
        # Post-RNN MLP layers
        x = rnn_out_squeezed
        for i, layer in enumerate(net.post_rnn):
            if isinstance(layer, nn.Linear):
                pre_act = x.detach()
                x = layer(x)
                activations[f'post_rnn_{i}'] = (pre_act, x.detach())
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.Sigmoid):
                x = layer(x)
                # Activation functions don't have weights, but we track activations
            else:
                x = layer(x)
        
        # EPLHb layers
        if hasattr(net, 'eplhb'):
            q_out = x.detach()
            eplhb_in = net.eplhb_input_norm(q_out)
            x_eplhb = eplhb_in
            for i, layer in enumerate(net.eplhb):
                if isinstance(layer, nn.Linear):
                    pre_act = x_eplhb.detach()
                    x_eplhb = layer(x_eplhb)
                    activations[f'eplhb_{i}'] = (pre_act, x_eplhb.detach())
                elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.Sigmoid):
                    x_eplhb = layer(x_eplhb)
                else:
                    x_eplhb = layer(x_eplhb)
        
        return activations

    def _apply_stdp_updates(self, net, activations, reward_signal):
        """
        Apply reward-modulated STDP updates to all network weights.
        
        STDP rule: Δw = η * reward_signal * (pre * post - decay * w)
        where pre is pre-synaptic activity, post is post-synaptic activity
        Uses learning_rate for main network, eplhb_lr for EPLHb layers
        """
        with th.no_grad():
            # Get current learning rate (handle Schedule objects)
            if isinstance(self.learning_rate, (float, int)):
                main_lr = self.learning_rate
            else:
                main_lr = self.learning_rate(self._current_progress_remaining)
            
            # Average reward signal across batch
            reward_magnitude = reward_signal.item() if isinstance(reward_signal, th.Tensor) else reward_signal
            reward_magnitude = abs(reward_magnitude)  # Use absolute value for magnitude
            
            # Update input projection
            if 'input_projection' in activations:
                pre, post = activations['input_projection']
                if pre is not None and hasattr(net, 'input_projection'):
                    # Compute update: pre (B, feat_dim) @ post (B, rnn_input_size)
                    # For linear layer: pre @ W = post, so W update should consider pre and post
                    pre_mean = pre.mean(dim=0, keepdim=True)  # [1, feat_dim]
                    post_mean = post.mean(dim=0, keepdim=True)  # [1, rnn_input_size]
                    update = pre_mean.t() @ post_mean  # [feat_dim, rnn_input_size]
                    net.input_projection.weight.data += (
                        main_lr * reward_magnitude * update
                        - self.stdp_decay * net.input_projection.weight.data
                    )
                    if net.input_projection.bias is not None:
                        net.input_projection.bias.data += (
                            main_lr * reward_magnitude * post_mean.squeeze()
                            - self.stdp_decay * net.input_projection.bias.data
                        )
            
            # Update RNN weights
            if 'rnn' in activations and hasattr(net, 'rnn'):
                pre, post = activations['rnn']
                if pre is not None:
                    pre_mean = pre.mean(dim=0, keepdim=True)  # [1, rnn_input_size]
                    post_mean = post.mean(dim=0, keepdim=True)  # [1, rnn_hidden_size]
                    
                    # Update input-to-hidden weights
                    if hasattr(net.rnn, 'weight_ih_l0'):
                        update_ih = pre_mean.t() @ post_mean  # [rnn_input_size, rnn_hidden_size]
                        net.rnn.weight_ih_l0.data += (
                            main_lr * reward_magnitude * update_ih
                            - self.stdp_decay * net.rnn.weight_ih_l0.data
                        )
                    
                    # Update hidden-to-hidden weights (use previous hidden state)
                    if hasattr(net.rnn, 'weight_hh_l0') and net._h is not None:
                        h_prev = net._h[0].mean(dim=0, keepdim=True)  # [1, rnn_hidden_size]
                        update_hh = h_prev.t() @ post_mean  # [rnn_hidden_size, rnn_hidden_size]
                        net.rnn.weight_hh_l0.data += (
                            main_lr * reward_magnitude * update_hh
                            - self.stdp_decay * net.rnn.weight_hh_l0.data
                        )
                    
                    # Update biases
                    if hasattr(net.rnn, 'bias_ih_l0') and net.rnn.bias_ih_l0 is not None:
                        net.rnn.bias_ih_l0.data += (
                            main_lr * reward_magnitude * post_mean.squeeze()
                            - self.stdp_decay * net.rnn.bias_ih_l0.data
                        )
            
            # Update post-RNN MLP layers
            for i, layer in enumerate(net.post_rnn):
                if isinstance(layer, nn.Linear):
                    key = f'post_rnn_{i}'
                    if key in activations:
                        pre, post = activations[key]
                        if pre is not None:
                            pre_mean = pre.mean(dim=0, keepdim=True)  # [1, in_features]
                            post_mean = post.mean(dim=0, keepdim=True)  # [1, out_features]
                            update = pre_mean.t() @ post_mean  # [in_features, out_features]
                            layer.weight.data += (
                                main_lr * reward_magnitude * update
                                - self.stdp_decay * layer.weight.data
                            )
                            if layer.bias is not None:
                                layer.bias.data += (
                                    main_lr * reward_magnitude * post_mean.squeeze()
                                    - self.stdp_decay * layer.bias.data
                                )
            
            # Update EPLHb layers (use eplhb_lr)
            if hasattr(net, 'eplhb'):
                for i, layer in enumerate(net.eplhb):
                    if isinstance(layer, nn.Linear):
                        key = f'eplhb_{i}'
                        if key in activations:
                            pre, post = activations[key]
                            if pre is not None:
                                pre_mean = pre.mean(dim=0, keepdim=True)  # [1, in_features]
                                post_mean = post.mean(dim=0, keepdim=True)  # [1, out_features]
                                update = pre_mean.t() @ post_mean  # [in_features, out_features]
                                layer.weight.data += (
                                    self.eplhb_lr * reward_magnitude * update
                                    - self.stdp_decay * layer.weight.data
                                )
                                if layer.bias is not None:
                                    layer.bias.data += (
                                        self.eplhb_lr * reward_magnitude * post_mean.squeeze()
                                        - self.stdp_decay * layer.bias.data
                                    )
            
            # Apply sign constraints if needed
            if self.policy_kwargs.get("fixed_sign", False):
                net.enforce_signs()

    def _train_recurrent(self, gradient_steps: int, batch_size: int = None, seq_len: int = None, batch_idxs: List[int] = None):
        losses = []
        net    = self.policy.q_net
        tgt    = self.q_net_target
        gamma  = self.gamma
        optim  = self.policy.optimizer

        l2_lambda = getattr(self, 'l2_lambda', None)
        if l2_lambda is None:
            # Try to get from policy_kwargs or pid_params
            l2_lambda = getattr(self, 'policy_kwargs', {}).get('l2_lambda', 1e-4)

        for _ in range(gradient_steps):
            # 1) sample a sequence
            if batch_idxs is not None:
                batch = self.replay_buffer.sample(
                    batch_idxs=batch_idxs,
                    env=self._vec_normalize_env,
                    seq_len=seq_len,
                )
            else:
                batch = self.replay_buffer.sample(
                    batch_size=batch_size,
                    env=self._vec_normalize_env,
                    seq_len=seq_len,
                )
            obs_seq      = batch.observations          # [B, L, obs_dim]
            act_seq      = batch.actions.squeeze(-1)   # [B, L]
            rew_seq      = batch.rewards.squeeze(-1)   # [B, L]
            done_seq     = batch.dones.squeeze(-1)     # [B, L]
            next_obs_seq = batch.next_observations     # [B, L, obs_dim]

            B, L, obs_dim = obs_seq.shape

            # 2) derivative term d_seq as in non‐recurrent .train()
            if self.tabular_d:
                d_seq = batch.ds.squeeze(-1)            # [B, L]
            else:
                obs_flat = obs_seq.reshape(B * L, obs_dim)
                act_flat = act_seq.reshape(B * L).long()
                with th.no_grad():
                    d_flat = self.d_net(obs_flat)
                    d_flat = d_flat.gather(1, act_flat.unsqueeze(1)).squeeze(1)
                d_seq = d_flat.reshape(B, L)

            # 3) initialize integrator from buffer
            z_prev = batch.zs.squeeze(-1)[:, 0]         # [B]

            # 4) reset RNN hidden state
            hidden = net.reset_hidden(batch_size=B, device=self.device)

            # 5) unroll L steps, compute Q‐predictions & PID targets
            q_pred_seq = []
            target_seq = []
            eplhb_seq  = []
            d_update_seq = []  # Store d_update values for auxiliary loss

            for t in range(L):
                # current Q
                q_t, _, eplhb_t = net.forward_full(obs_seq[:, t, :])
                a_t  = act_seq[:, t].unsqueeze(1)        # [B, 1]
                q_at = q_t.gather(1, a_t).squeeze(1)     # [B]

                # store eplhb output for later
                eplhb_seq.append(eplhb_t.unsqueeze(1)) # [B, 1]

                with th.no_grad():
                    # 1‐step Bellman target
                    q_tp1 = tgt(next_obs_seq[:, t, :]).max(dim=1)[0]
                    td_target = rew_seq[:, t] + (1 - done_seq[:, t]) * gamma * q_tp1

                    # PID gains & smoothers
                    kp, ki, kd, alpha, beta = self.gain_adapter.get_gains(
                        obs_seq[:, t, :], a_t, batch
                    )

                    BR    = td_target - q_at                     # [B]
                    z_new = beta * z_prev + alpha * BR    # [B]

                    p_up  = BR
                    i_up  = z_new
                    d_up  = q_at - d_seq[:, t]            # [B]

                    # Base target without EPLHb (computed outside no_grad)
                    base_target_t = q_at + kp * p_up + ki * i_up + kd * d_up  # [B]

                # Add EPLHb contribution to target (this allows gradients to flow)
                # print(f"net.eplhb_coeff * eplhb_t: {net.eplhb_coeff * eplhb_t}")
                target_t = base_target_t + net.eplhb_coeff * eplhb_t  # [B]
                # target_t = base_target_t

                q_pred_seq.append(q_at.unsqueeze(1))     # list of [B,1]
                target_seq.append(target_t.unsqueeze(1)) # list of [B,1]
                d_update_seq.append(d_up.unsqueeze(1))   # list of [B,1] - store for auxiliary loss
                z_prev = z_new

                self.p_update = BR
                self.d_update = d_up
                self.i_update = i_up

                # Update the gains
                self.kp = kp
                self.ki = ki
                self.kd = kd


            # 6) one‐shot loss over full sequence
            q_pred = th.cat(q_pred_seq, dim=1)         # [B, L]
            target = th.cat(target_seq, dim=1)         # [B, L]
            target = target.view_as(q_pred)   # reshape target to exactly q_pred's shape
        
            # Concatenate EPLHb outputs and d_update values for auxiliary loss
            eplhb_concat = th.cat(eplhb_seq, dim=1)    # [B, L]
            d_update_concat = th.cat(d_update_seq, dim=1)  # [B, L]
            
            # 6) one-shot loss over full sequence
            # q_pred_seq and target_seq are [B, L], so:
            td_error_seq = q_pred - target  # shape [B, L]
            base_loss = F.smooth_l1_loss(q_pred, target) # [B, L]
            noise_seq = th.randn_like(td_error_seq)

            # Auxiliary loss: EPLHb should be similar to d_update value
            d_update_loss = F.smooth_l1_loss(eplhb_concat, d_update_concat)

            # L2 regularization for EPLHb weights
            l2_penalty = 0.0
            if hasattr(self.policy.q_net, 'eplhb') and hasattr(self.policy.q_net.eplhb[0], 'weight') and l2_lambda != 0:
                l2_penalty = l2_lambda * th.sum(self.policy.q_net.eplhb[0].weight ** 2)

            final_loss = (
                 base_loss
                #  + (noise_seq * td_error_seq).mean()
                + l2_penalty
            )

            if self.use_stdp:
                # STDP-based training: use reward-modulated plasticity
                # Reset hidden state for activation capture
                net.reset_hidden(batch_size=B, device=self.device)
                
                # Capture activations for the entire sequence
                # Average across sequence length for reward signal
                # Use mean TD error as reward signal (inverted, so lower error = higher reward)
                td_error_magnitude = td_error_seq.abs().mean()
                reward_signal = -td_error_magnitude  # Negative TD error as reward signal
                
                # Capture activations for each timestep and accumulate updates
                all_activations = []
                for t in range(L):
                    # Forward pass to capture activations
                    obs_t = obs_seq[:, t, :]
                    act_t = act_seq[:, t].unsqueeze(1)
                    
                    # Capture activations
                    activations = self._capture_activations(net, obs_t, act_t)
                    all_activations.append(activations)
                
                # Apply STDP updates using average activations across sequence
                # Use mean TD error as reward signal (inverted, so lower error = higher reward)
                # td_error_magnitude = td_error_seq.abs().mean()
                # reward_signal = -td_error_magnitude  # Negative TD error as reward signal
                
                # Average activations across timesteps
                avg_activations = {}
                if len(all_activations) > 0:
                    for key in all_activations[0].keys():
                        pre_list = [act[key][0] for act in all_activations if act[key][0] is not None]
                        post_list = [act[key][1] for act in all_activations if act[key][1] is not None]
                        
                        if len(pre_list) > 0:
                            avg_pre = th.stack(pre_list).mean(dim=0)
                        else:
                            avg_pre = None
                        
                        if len(post_list) > 0:
                            avg_post = th.stack(post_list).mean(dim=0)
                        else:
                            avg_post = None
                        
                        avg_activations[key] = (avg_pre, avg_post)
                
                # Apply STDP updates
                if len(avg_activations) > 0:
                    self._apply_stdp_updates(net, avg_activations, reward_signal)
                
                # Also update EPLHb coeff if needed (using simple gradient-like update)
                if hasattr(net, 'eplhb_coeff_raw'):
                    # Simple update: increase coeff if EPLHb helps reduce loss
                    with th.no_grad():
                        coeff_update = -self.eplhb_lr * d_update_loss.mean()
                        net.eplhb_coeff_raw.data += coeff_update
                        # Clip to reasonable range
                        net.eplhb_coeff_raw.data.clamp_(-10.0, 10.0)
                
            else:
                # Standard backpropagation training
                optim.zero_grad()
                
                # Compute gradients for main network (excluding EPLHb parameters)
                final_loss.backward(retain_graph=True)
                
                # Compute gradients for EPLHb network only
                if hasattr(self.policy.q_net, 'eplhb'):
                    # Get EPLHb parameters
                    eplhb_params = list(self.policy.q_net.eplhb.parameters())
                    if hasattr(self.policy.q_net, 'eplhb_coeff_raw'):
                        eplhb_params.append(self.policy.q_net.eplhb_coeff_raw)
                    
                    # Compute gradients only for EPLHb parameters
                    eplhb_grads = th.autograd.grad(
                        d_update_loss, eplhb_params, retain_graph=True, create_graph=False, allow_unused=True
                    )
                    
                    # Manually set gradients for EPLHb parameters
                    for param, grad in zip(eplhb_params, eplhb_grads):
                        if grad is not None:  # Only update if gradient was computed
                            if param.grad is not None:
                                param.grad += grad
                            else:
                                param.grad = grad
                    
                    # Apply gradient clipping to EPLHb parameters
                    th.nn.utils.clip_grad_norm_(eplhb_params, 1.0)
                    if hasattr(self.policy.q_net, 'eplhb_coeff_raw'):
                        th.nn.utils.clip_grad_norm_([self.policy.q_net.eplhb_coeff_raw], 0.1)

                # Apply gradient clipping to all parameters
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                # Update all parameters with the optimizer
                optim.step()
                
            if self.policy_kwargs.get("fixed_sign", False):
                self.policy.q_net.enforce_signs()
                self.policy.q_net_target.enforce_signs()

            losses.append(final_loss.item())

        # 7) logging (same as non‐recurrent)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss",      np.mean(losses))
        self.logger.dump(step=self.num_timesteps)


    def compute_next_q_values(self, replay_data, batch_size):
        with th.no_grad():
            if self.policy_evaluation:
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_actions = th.tensor(self.optimal_model.predict(replay_data.next_observations.cpu())[0]).reshape(-1,1)
                next_actions = next_actions.to(next_q_values.device)
                next_q_values = th.gather(next_q_values, dim=1, index=next_actions)
            elif self.is_double:
                # Double DQN
                next_q_values = self.q_net(replay_data.next_observations)
                next_actions = th.argmax(next_q_values, dim=1)
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_values[range(batch_size), next_actions]
            else:
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)

            return next_q_values.reshape(-1, 1)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if self.policy_evaluation:
            action = self.optimal_model.predict(observation)[0]
            return action, state
        
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)

        if self.dump_buffer:
            # Calling self.monte_carlo_rollout() 1 time to build the Q value buffer is enough as the environments we deal with are deterministic
            true_q_value = self.monte_carlo_rollout(action)
            self.buffer.append((*observation, action, true_q_value))

        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        outputs = super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        if self.dump_buffer:
            np.save(f"{globals.base_directory}/models/{self.visualization_env.unwrapped.spec.id}/bufferQValues.npy", self.buffer)

        return outputs

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def visualize_episode(self, file_name="episode", max_length=10000):
        """Render the environment until the episode is done.

        Args:
            file_name (str, optional): The name of the file. Defaults to "episode".
        """
        env = RecordVideo(self.visualization_env, file_name + ".mp4")

        state = env.reset()[0]
        done = False
        k = 0

        while not done and k < max_length:
            # Take an action
            action = self.predict(state, deterministic=True)[0]
            # Take the action
            state, _, done, _, _ = env.step(action)
            k += 1

        env.close()

    def monte_carlo_rollout(self, action):
        return run_simulation(self.optimal_model, deepcopy(self.env.envs[0]), action, self.gamma, self.seed)


class PID_DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from the Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self, d_tau, tabular_d,
        gain_adapter,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        dump_buffer: bool = False,
        is_double=False,
        optimal_model=None,
        policy_evaluation=False
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True
        )
        # The stable baselines wrapped env don't play nice with the RecordVideo Wrapper
        # The simplest solution is to reserve an unwrapped instance for video recording, instead of modifying the API
        # Our additions atop stable baselines:
        self.visualization_env = env
        self.d_tau = d_tau
        self.tabular_d = tabular_d
        self.dump_buffer = dump_buffer
        self.buffer = []  # The buffer we dump, if dump_buffer is True
        self.is_double = is_double
        self.optimal_model = optimal_model
        self.policy_evaluation = policy_evaluation

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0


        if _init_setup_model:
            self._setup_model()

        # Gain adaptation Code
        self.gain_adapter = gain_adapter
        self.BRs = None
        self.previous_p_update, self.p_update = None, None
        self.previous_i_update, self.i_update = None, None
        self.previous_d_update, self.d_update = None, None
        self.kp, self.ki, self.kd = None, None, None
        self.full_p_update, self.full_i_update, self.full_d_update = None, None, None
        self.full_kp, self.full_ki, self.full_kd = None, None, None

        self.policy.jump_start_cuda()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target
        self.d_net = self.policy.d_net

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            # Update the gains here once we have started training
            if self._n_calls > self.learning_starts:
                #self.gain_adapter.apply_weight_decay(self.replay_buffer)
                update_size = 50000
                update_size = min(update_size, self.replay_buffer.size())

                replay_data = self.replay_buffer.sample(
                        batch_idxs=[0],
                        env=self._vec_normalize_env,
                    )

                self.gain_adapter.adapt_gains(replay_data)

            # Update the D network
            polyak_update(self.q_net_target.parameters(), self.d_net.parameters(), self.d_tau)
            # Update the target network
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_idxs: int = 0) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        # If we have an RNN, do truncated BPTT
        is_recurrent = hasattr(self.policy.q_net, "reset_hidden")
        use_bptt   = is_recurrent
        if use_bptt:
            return self._train_recurrent(gradient_steps, batch_idxs)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                next_q_values = self.compute_next_q_values(replay_data, len(batch_idxs))
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Current Q estimates of the target network
                target_current_q_values = self.q_net_target(replay_data.observations)
                target_current_q_values = th.gather(target_current_q_values, dim=1, index=replay_data.actions.long())

                if self.tabular_d:
                    d_values = replay_data.ds
                else:
                    d_values = self.d_net(replay_data.observations)
                    d_values = th.gather(d_values, dim=1, index=replay_data.actions.long())

                kp, ki, kd, alpha, beta = self.gain_adapter.get_gains(
                    replay_data.observations, replay_data.actions, replay_data
                )
                self.BRs = target_q_values - target_current_q_values
                new_zs = beta * replay_data.zs + alpha * self.BRs

                self.p_update = self.BRs
                self.d_update = target_current_q_values - d_values
                self.i_update = new_zs

                # Update the gains
                self.kp = kp
                self.ki = ki
                self.kd = kd

                target = target_current_q_values + kp * self.p_update + ki * self.i_update + kd * self.d_update

            # if np.random.rand() < 0.005:
            #     breakpoint()

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("rollout/BRs", th.mean(self.BRs).item())
        self.logger.record("rollout/BRs_L2_norm", th.norm(self.BRs).item() / np.sqrt(self.BRs.shape[0]))
        self.logger.dump(step=self.num_timesteps)

    def _train_recurrent(self, gradient_steps: int, batch_idxs: List[int]) -> None:
        losses = []
        net    = self.policy.q_net
        tgt    = self.q_net_target
        gamma  = self.gamma
        optim  = self.policy.optimizer

        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(
                batch_idxs=batch_idxs,
                env=self._vec_normalize_env,
            )

            

            obs_seq      = batch.observations          # [B, L, obs_dim]
            next_obs_seq = batch.next_observations     # [B, L, obs_dim]

            # New: keep that last dim as your time axis
            act_seq  = batch.actions     # [B, L]
            rew_seq  = batch.rewards     # [B, L]
            done_seq = batch.dones

            B, L, obs_dim = obs_seq.shape

            # 2) derivative term d_seq as in non‐recurrent .train()
            if self.tabular_d:
                d_seq = batch.ds.squeeze(-1)            # [B, L]
            else:
                obs_flat = obs_seq.reshape(B * L, obs_dim)
                act_flat = act_seq.reshape(B * L).long()
                with th.no_grad():
                    d_flat = self.d_net(obs_flat)
                    d_flat = d_flat.gather(1, act_flat.unsqueeze(1)).squeeze(1)
                d_seq = d_flat.reshape(B, L)

            # 3) initialize integrator from buffer
            z_prev = batch.zs[:, 0]         # [B]

            # 4) reset RNN hidden state
            net.reset_hidden(batch_size=B, device=self.device)

            # 5) unroll L steps, compute Q‐predictions & PID targets
            q_pred_seq = []
            target_seq = []

            for t in range(L):
                # current Q
                q_t  = net(obs_seq[:, t, :])            # [B, n_actions]
                a_t  = act_seq[:, t].unsqueeze(1)        # [B, 1]
                q_at = q_t.gather(1, a_t).squeeze(1)     # [B]

                with th.no_grad():
                    # 1‐step Bellman target
                    q_tp1 = tgt(next_obs_seq[:, t, :]).max(dim=1)[0]
                    td    = rew_seq[:, t] + (1 - done_seq[:, t]) * gamma * q_tp1

                    # PID gains & smoothers
                    kp, ki, kd, alpha, beta = self.gain_adapter.get_gains(
                        obs_seq[:, t, :], a_t, batch
                    )

                    BR    = td - q_at                     # [B]
                    z_new = beta * z_prev + alpha * BR    # [B]

                    p_up  = BR
                    i_up  = z_new
                    d_up  = q_at - d_seq[:, t]            # [B]

                    target_t = q_at + kp * p_up + ki * i_up + kd * d_up  # [B]

                q_pred_seq.append(q_at.view(-1, 1))    # list of [B,1]
                target_seq.append(target_t.view(-1, 1)) # list of [B,1]
                z_prev = z_new

                self.p_update = BR
                self.d_update = d_up
                self.i_update = i_up

                # Update the gains
                self.kp = kp
                self.ki = ki
                self.kd = kd


            # 6) one‐shot loss over full sequence
            q_pred = th.cat(q_pred_seq, dim=1)         # [B, L]
            target = th.cat(target_seq, dim=1)         # [B, L]
            target = target.view_as(q_pred)   # reshape target to exactly q_pred’s shape

            loss   = F.smooth_l1_loss(q_pred, target)
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            optim.step()
            if self.policy_kwargs["fixed_sign"]:
                self.policy.q_net.enforce_signs()
                self.policy.q_net_target.enforce_signs()

        # 7) logging (same as non‐recurrent)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss",      np.mean(losses))
        self.logger.dump(step=self.num_timesteps)

    def compute_next_q_values(self, replay_data, batch_size):
        with th.no_grad():
            if self.policy_evaluation:
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_actions = th.tensor(self.optimal_model.predict(replay_data.next_observations.cpu())[0]).reshape(-1,1)
                next_actions = next_actions.to(next_q_values.device)
                next_q_values = th.gather(next_q_values, dim=1, index=next_actions)
            elif self.is_double:
                # Double DQN
                next_q_values = self.q_net(replay_data.next_observations)
                next_actions = th.argmax(next_q_values, dim=1)
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_values[range(batch_size), next_actions]
            else:
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)

            return next_q_values.reshape(-1, 1)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if self.policy_evaluation:
            action = self.optimal_model.predict(observation)[0]
            return action, state

        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)

        if self.dump_buffer:
            # Calling self.monte_carlo_rollout() 1 time to build the Q value buffer is enough as the environments we deal with are deterministic
            true_q_value = self.monte_carlo_rollout(action)
            self.buffer.append((*observation, action, true_q_value))

        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        outputs = super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        if self.dump_buffer:
            np.save(f"{globals.base_directory}/models/{self.visualization_env.unwrapped.spec.id}/bufferQValues.npy", self.buffer)

        return outputs

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def visualize_episode(self, file_name="episode", max_length=10000):
        """Render the environment until the episode is done.

        Args:
            file_name (str, optional): The name of the file. Defaults to "episode".
        """
        env = RecordVideo(self.visualization_env, file_name + ".mp4")

        state = env.reset()[0]
        done = False
        k = 0

        while not done and k < max_length:
            # Take an action
            action = self.predict(state, deterministic=True)[0]
            # Take the action
            state, _, done, _, _ = env.step(action)
            k += 1

        env.close()

    def monte_carlo_rollout(self, action):
        return run_simulation(self.optimal_model, deepcopy(self.env.envs[0]), action, self.gamma, self.seed)