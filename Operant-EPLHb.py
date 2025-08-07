#!/usr/bin/env python3
import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
# from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN # not working for me
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer
from TabularPID.Agents.DQN.DQN import EPLHb_DQN, PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

from OperantGym import OperantLearning
from plotfunctions import plot_figure
from recorder import SessionRecorder

# --------------------
# Hyperparameters
# --------------------
session_params = {
    "pairing":          'reward',
    "num_trials":       200,
    "pre_steps":        10,           # 1 s @ 100 ms
    "post_steps":       40,           # 5 s @ 100 ms
    "enl_duration":     (2.0, 4.0),   # seconds
    "tau_on":           0.01,         # 10 ms
    "tau_off":          0.1,          # 100 ms

    "omission_prob":    0.05,
    "action_cost":      0.1,
    "enl_penalty":      0.2,
    "enl_threshold":    200,          # for accumulated & consecutive ENL licks
    "enl_punish_scale": 0.1,

    "gradient_steps":   10,           # how many rollout‐training steps per trial
    "gamma":            0.95,         # discount factor
    "batch_training":   False,
    "batch_size":       64 if False else 1,
    "buffer_size":      100_000 if False else 1,
    "dt":               0.1,          # 100 ms
}


# PID-DQN parameters
pid_params = {
    "learning_rate": 1e-3,
    "eplhb_lr": 1e-2,
    "coeff_lr": 0.0,
    "initial_eplhb_coeff": -0.0,

    "rnn_type": "GRU",  # Options: "RNN", "GRU", "LSTM". Change as needed.
     "l2_lambda": 1e-7,  # L2 regularization strength for EPLHb weights

    "replay_memory_size": session_params["buffer_size"],
    "batch_size": session_params["batch_size"],
    "tau": 1,
    "gamma": session_params["gamma"],
    "gradient_steps": 1,
    "train_freq": 1,
    "target_update_interval": 10,
    "initial_eps": 0.1,
    "exploration_fraction": 0.001,
    "minimum_eps": 0.05,
    "learning_starts": 1000,
    "inner_size": 64,
    "dump_buffer": False,
    "is_double": False,
    "policy_evaluation": False,
    "seed": 12242,

    "kp": 1.0,
    "ki": 0.0,
    "kd": 0.0,
    "meta_lr": 0,
    "epsilon_gain": 0.1,
    "alpha": 0.05,
    "beta": 0.95,
    "d_tau": 1,
    "tabular_d": False,
}

# Other params
replaybuffer = OnlineReplayBuffer
max_trial_steps = session_params["pre_steps"] + session_params["post_steps"]

# --------------------
# Setup
# --------------------
env = OperantLearning(
    pairing=session_params["pairing"],
    omission_prob=session_params["omission_prob"],
    enl_duration=session_params["enl_duration"],
    action_cost=session_params["action_cost"],
    enl_penalty=session_params["enl_penalty"],
    reward_decay=True,
    reward_decay_time=2.0,
    print_status=False,
)

# Gain adapter
gain_adapter = SingleGainAdapter(
    kp=pid_params["kp"],
    ki=pid_params["ki"],
    kd=pid_params["kd"],
    alpha=pid_params["alpha"],
    beta=pid_params["beta"],
    meta_lr=pid_params["meta_lr"],
    epsilon=pid_params["epsilon_gain"],
)

# Define policy kwargs (network architecture + optimizer)
policy_kwargs = dict(
    net_arch=[pid_params["inner_size"], pid_params["inner_size"]],
    optimizer_class=optim.Adam,
    with_RNN_layer=True,
    rnn_type=pid_params["rnn_type"],  # Options: "RNN", "GRU", "LSTM". Change as needed.
    features_extractor_kwargs=dict(
        initial_eplhb_coeff=pid_params["initial_eplhb_coeff"],  # <-- Set your desired initial value here
    ),
)

# EPLHb-specific optimizer kwargs
optimizer_kwargs = dict(
    eplhb_lr=pid_params["eplhb_lr"],   # your custom learning rate for EPLHb layer
    coeff_lr=pid_params["coeff_lr"],   # your custom learning rate for eplhb_coeff
    # ... any other optimizer kwargs ...
)

# Prevent CUDA from being used (patch)
import TabularPID.Agents.DQN.DQN_policy as _dp
# find whatever class has jump_start_cuda and override it
for _name in dir(_dp):
    cls = getattr(_dp, _name)
    if isinstance(cls, type) and hasattr(cls, "jump_start_cuda"):
        cls.jump_start_cuda = lambda self: None


# Set up the model
model = EPLHb_DQN(
    # PID-specific arguments
    pid_params['d_tau'],
    pid_params['tabular_d'],
    gain_adapter,

    # standard SB3/DQN args
    policy="MlpPolicy",
    env=env,
    learning_rate=pid_params['learning_rate'],
    buffer_size=pid_params['replay_memory_size'],
    batch_size=pid_params['batch_size'],
    tau=pid_params['tau'],
    gamma=pid_params['gamma'],
    gradient_steps=pid_params['gradient_steps'],
    train_freq=pid_params['train_freq'],
    target_update_interval=pid_params['target_update_interval'],
    exploration_fraction=pid_params['exploration_fraction'],
    exploration_initial_eps=pid_params['initial_eps'],
    exploration_final_eps=pid_params['minimum_eps'],
    optimize_memory_usage=False,
    learning_starts=pid_params['learning_starts'],
    tensorboard_log=None,
    policy_kwargs=policy_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    seed=pid_params['seed'],
    device="cpu",
    dump_buffer=pid_params['dump_buffer'],
    is_double=pid_params['is_double'],
    optimal_model=None,
    policy_evaluation=pid_params['policy_evaluation'],

    # Use authors replay buffer
    replay_buffer_class=replaybuffer,
)
orig_buffer = model.replay_buffer # save the original big replay buffer
model.l2_lambda = pid_params["l2_lambda"]  # Set L2 regularization for EPLHb weights

# Link adapter to model
gain_adapter.set_model(model)

# Set up logging
new_logger = configure(None, ["stdout"])
model.set_logger(new_logger)
recorder = SessionRecorder()

# epsilon decay params
eps_start   = pid_params["initial_eps"]
eps_end     = pid_params["minimum_eps"]
max_num_iters = 40000
decay_trials = int(pid_params["exploration_fraction"] * max_num_iters)

# Set up buffer
model.replay_buffer = OnlineReplayBuffer(
    buffer_size=10_000, # hold the last 10 000 steps (ie 1000 seconds)
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=model.device,
    optimize_memory_usage=False,
    handle_timeout_termination=True,
)

# --------------------
# Training Loop
# --------------------

# Unpack some session parameters
num_trials = session_params["num_trials"]
batch_size = session_params["batch_size"]
enl_threshold = session_params["enl_threshold"]
enl_punish_scale = session_params["enl_punish_scale"]
gradient_steps = session_params["gradient_steps"]

obs, _ = env.reset()
trial_idx = 0
eps = pid_params["initial_eps"]
# — prime the recorder so rec._prev_obs isn't None on step 0 —
recorder._prev_obs = obs

while trial_idx < num_trials:
    print(f"Trial {trial_idx+1}/{num_trials}, ε={eps:.3f}")

    # reset RNN state
    model.policy.q_net.reset_hidden(batch_size=session_params["batch_size"])
    done = False
    trial_timesteps = 0
    enl_count = 0
    z_prev = 0.0
    
    # run one trial
    while not done:
        # Set exploration rate
        model.exploration_rate = eps
        model.logger.record("rollout/exploration_rate", eps)

        # act
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, _, _, info = env.step(action)
        done = info["done"]
        outcome = info["outcome"]

        # punish if stuck in ENL for > 200 steps
        enl_count = enl_count + 1 if outcome and "enl" in outcome else 0
        reward -= max(enl_count - enl_threshold, 0) * enl_punish_scale

        # update gains and sync networks
        model._on_step()
        trial_timesteps += 1
        
        # calculate d and z updates for replay buffer
        with torch.no_grad():
            # make observation tensor
            obs_t = torch.tensor(obs, device=model.device, dtype=torch.float32).unsqueeze(0) # [1, obs_dim]
            next_t = torch.tensor(next_obs, device=model.device, dtype=torch.float32).unsqueeze(0) # [1, obs_dim]
            
            # get the D update
            if model.tabular_d:
                d_update = model.gain_adapter.get_d_update(obs_t, next_t)
            else:
                d_out = model.d_net(obs_t) # [1, n_actions]
                d_update = d_out[0, action].item()  # get the D update for the action taken
            
            # get your PID gains α, β (and kp,ki,kd if you want)
            action_scalar = int(action)  
            a_t = torch.tensor([[action_scalar]], dtype=torch.long, device=model.device)
            kp, ki, kd, alpha, beta = model.gain_adapter.get_gains(obs_t, a_t, None)
            # print("Gains:", kp.item(), ki.item(), kd.item())
            
            # d) Q-values for TD‐error
            q_curr = model.policy.q_net(obs_t)[0, action].item()
            q_next = model.policy.q_net_target(next_t).max(dim=1)[0].item()
            td_err = reward + (0.0 if done else model.gamma * q_next) - q_curr  # BRₜ
            # print("Q current:", q_curr, "Q next:", q_next, "TD Error:", td_err, "Reward:", reward, "Done:", done, "Action:", action_scalar)

            # # e) integrator update
            z_update = beta * z_prev + alpha * td_err

            # By calling q_reward = q_net(next_t) you explicitly feed the online net the next observation 
            # and let its hidden state advance to reflect that transition.
            # Without that, the hidden state of the online net never "sees" the reward‐state until the following time step, 
            # so your RNN is perpetually one step behind. Over many steps—especially in those ENL‐stuck trials—that misalignment can cause it to keep choosing the same action forever.
            q_cue    = model.policy.q_net(obs_t)[0, action_scalar].item()
            q_reward = model.policy.q_net(next_t)[0, action_scalar].item()
        
        # add to the replay buffer
        model.replay_buffer.add(obs=obs, next_obs=next_obs,
                                action=np.array([action]),
                                reward=np.array([reward], dtype=np.float32),
                                done=done, infos=[info],
                                d=np.array([d_update], dtype=np.float32), 
                                z=np.array([z_update], dtype=np.float32),
                                )
        
        # record every timestep in the session trace
        recorder.record_env_step(trial_idx, action, reward, next_obs, info, model=model,
                                record_sign_index=True, record_eplhb_weight=True)
        
        # # Debug: Print EPLHb information every 10 trials
        # if trial_idx % 10 == 0:
        #     with torch.no_grad():
        #         obs_t = torch.tensor(obs, device=model.device, dtype=torch.float32).unsqueeze(0)
        #         q_pred, _, eplhb_out = model.policy.q_net.forward_full(obs_t)
        #         eplhb_contribution = model.policy.q_net.eplhb_coeff * eplhb_out
        #         q_value = q_pred[0, action].item()
                
        #         # Check EPLHb network weights
        #         eplhb_weights = list(model.policy.q_net.eplhb.parameters())
        #         weight_norm = eplhb_weights[0].norm().item() if len(eplhb_weights) > 0 else 0
                
        #         # Check EPLHb gradients if they exist
        #         if len(eplhb_weights) > 0 and eplhb_weights[0].grad is not None:
        #             grad_norm = eplhb_weights[0].grad.norm().item()
        #             print(f"EPLHb gradient norm = {grad_norm:.6f}")
                
        #         # Print first few weights
        #         if len(eplhb_weights) > 0:
        #             first_weights = eplhb_weights[0].data.flatten()[:5].cpu().numpy()
        #             print(f"First 5 EPLHb weights: {first_weights}")
                
        #         # Print optimizer learning rates
        #         optimizer = model.policy.optimizer
        #         print(f"Optimizer param groups:")
        #         for i, group in enumerate(optimizer.param_groups):
        #             print(f"  Group {i}: lr = {group['lr']:.6f}, params = {len(group['params'])}")
                
        #         # Get PID terms from the model
        #         if hasattr(model, 'p_update') and model.p_update is not None:
        #             p_term = model.p_update.item() if hasattr(model.p_update, 'item') else model.p_update
        #             kp_val = model.kp.item() if hasattr(model.kp, 'item') else model.kp
        #             pid_contribution = kp_val * p_term
                    
        #             print(f"Trial {trial_idx}: Q-value = {q_value:.4f}, "
        #                   f"EPLHb coeff = {model.policy.q_net.eplhb_coeff.item():.4f}, "
        #                   f"EPLHb output = {eplhb_out.item():.4f}, "
        #                   f"EPLHb contribution = {eplhb_contribution.item():.4f}, "
        #                   f"Relative contribution = {abs(eplhb_contribution.item()/q_value)*100:.2f}%, "
        #                   f"P term = {p_term:.4f}, kp = {kp_val:.4f}, "
        #                   f"PID contribution = {pid_contribution:.4f}, "
        #                   f"EPLHb/PID ratio = {abs(eplhb_contribution.item()/pid_contribution)*100:.2f}%, "
        #                   f"EPLHb weight norm = {weight_norm:.4f}")
        #         else:
        #             print(f"Trial {trial_idx}: Q-value = {q_value:.4f}, "
        #                   f"EPLHb coeff = {model.policy.q_net.eplhb_coeff.item():.4f}, "
        #                   f"EPLHb output = {eplhb_out.item():.4f}, "
        #                   f"EPLHb contribution = {eplhb_contribution.item():.4f}, "
        #                   f"Relative contribution = {abs(eplhb_contribution.item()/q_value)*100:.2f}%, "
        #                   f"EPLHb weight norm = {weight_norm:.4f}")
        
        # update obs, z_prev
        obs, z_prev = next_obs, z_update

    if outcome == "trial_end": 
        trial_idx += 1 # update trial index
        # compute step-based epsilon
        frac = min(1.0, trial_idx / max(1, decay_trials))
        eps  = eps_start + frac * (eps_end - eps_start)

    # 4) do a single training step
    model.train(batch_size=batch_size, seq_len=trial_timesteps, gradient_steps=gradient_steps)

    # 5) restore original buffer so you keep accumulating long-term experience
    model.replay_buffer = orig_buffer


# --------------------
# Plot Summary Figure
# --------------------
plot_figure(recorder, td_error_type='internal', dt=session_params["dt"], show=True,
            pre_steps=session_params["pre_steps"], post_steps=session_params["post_steps"])
