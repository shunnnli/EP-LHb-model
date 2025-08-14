#!/usr/bin/env python3
import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer
from TabularPID.Agents.DQN.DQN import EPLHb_DQN, PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from OperantGym import OperantLearning
from plotfunctions import plot_figure
from recorder import SessionRecorder
import matplotlib.pyplot as plt

seed = 12242

# ============================================================================
# OPERANT ENVIRONMENT PARAMETERS (exactly matching EPLHb-Operant.py)
# ============================================================================
operant_session_params = {
    "pairing":          'reward',
    "num_trials":       2,
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

    "dt":               0.1,          # 100 ms
}

operant_pid_params = {
    "learning_rate": 1e-3,
    "eplhb_lr": 1e-2,
    "coeff_lr": 0.0,
    "initial_eplhb_coeff": -0.3,

    "rnn_type": "GRU",  # Options: "RNN", "GRU", "LSTM"
    "l2_lambda": 0.0,  # L2 regularization strength for EPLHb weights

    "batch_training": False,
    "batch_size": 64 if False else 1,
    "buffer_size": 100_000 if False else 1,
    "tau": 1,
    "gamma": 0.95,
    "gradient_steps": 10,
    "train_freq": 1,
    "target_update_interval": 10,
    
    "initial_eps": 0.1,
    "exploration_fraction": 0.5,
    "minimum_eps": 0.05,
    "learning_starts": 1,
    "inner_size": 64,
    "dump_buffer": False,
    "is_double": False,
    "policy_evaluation": False,
    "seed": seed,

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

# ============================================================================
# GYM ENVIRONMENT PARAMETERS (exactly matching EPLHb-Gym.py)
# ============================================================================
gym_env_params = {
    "env_name": "CliffWalking-v0",  # Change to "CliffWalking-v0" or any Gymnasium env
    "num_episodes": 2,
    "max_steps": 500,
    "warmup_steps": 10000,
    "train_every_n_steps": 100,  # You can adjust this value
    "render_mode": "none",  # Use "human" for rendering, or None for no rendering
}

gym_pid_params = {
    "learning_rate": 1e-3,
    "eplhb_lr": 1e-2,
    "coeff_lr": 0.0,
    "initial_eplhb_coeff": -1.0,
    
    "rnn_type": "GRU",  # Options: "RNN", "GRU", "LSTM"
    "l2_lambda": 0.0,
    
    "buffer_size": 100_000,
    "batch_size": 1,
    "tau": 1,
    "gamma": 0.99,
    "gradient_steps": 10,
    "train_freq": 1,
    "target_update_interval": 10,
    
    "initial_eps": 1,
    "exploration_fraction": 0.8,
    "minimum_eps": 0.05,
    "learning_starts": 1000,
    "inner_size": 64,
    "dump_buffer": False,
    "is_double": False,
    "policy_evaluation": False,
    "seed": seed,

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

# ============================================================================
# TRANSFER LEARNING PARAMETERS
# ============================================================================
transfer_params = {
    "source_env": "operant",  # "operant" or "gym"
    "target_env": "gym",      # "operant" or "gym"
    "fix_source_weights": 10,
}

# ============================================================================
# ENVIRONMENT SELECTION AND SETUP
# ============================================================================
def setup_environment(env_type):
    """Setup environment based on type (operant or gym)"""
    if env_type == "operant":
        env = OperantLearning(
            pairing=operant_session_params["pairing"],
            omission_prob=operant_session_params["omission_prob"],
            enl_duration=operant_session_params["enl_duration"],
            action_cost=operant_session_params["action_cost"],
            enl_penalty=operant_session_params["enl_penalty"],
            reward_decay=True,
            reward_decay_time=1.0,
            print_status=False,
        )
        return env, operant_session_params, operant_pid_params
    elif env_type == "gym":
        env = gym.make(gym_env_params["env_name"], render_mode=gym_env_params["render_mode"])
        return env, gym_env_params, gym_pid_params
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

def setup_model(env, pid_params):
    """Setup EPLHb_DQN model with appropriate parameters"""
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
        rnn_type=pid_params["rnn_type"],
        features_extractor_kwargs=dict(
            initial_eplhb_coeff=pid_params["initial_eplhb_coeff"],
        ),
    )

        # EPLHb-specific optimizer kwargs
    optimizer_kwargs = dict(
        eplhb_lr=pid_params["eplhb_lr"],
        coeff_lr=pid_params["coeff_lr"],
    )

    # Prevent CUDA from being used (patch)
    import TabularPID.Agents.DQN.DQN_policy as _dp
    for _name in dir(_dp):
        cls = getattr(_dp, _name)
        if isinstance(cls, type) and hasattr(cls, "jump_start_cuda"):
            cls.jump_start_cuda = lambda self: None

        # Set up the model
    model = EPLHb_DQN(
            pid_params['d_tau'],
            pid_params['tabular_d'],
            gain_adapter,

        policy="MlpPolicy",
            env=env,
        learning_rate=pid_params['learning_rate'],
        buffer_size=pid_params['buffer_size'],
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

            replay_buffer_class=OnlineReplayBuffer,
    )
    
    # Set additional parameters
    if hasattr(model, 'l2_lambda'):
        model.l2_lambda = pid_params["l2_lambda"]
    
    # Link adapter to model
    gain_adapter.set_model(model)
    
    return model, gain_adapter

def setup_buffer(model, env_type, env, warmup_steps=10000):
    if env_type == "operant":
        orig_buffer = model.replay_buffer
        model.replay_buffer = OnlineReplayBuffer(
            buffer_size=10_000, # hold the last 10 000 steps (ie 1000 seconds)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=model.device,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
        )
        return orig_buffer
    elif env_type == "gym":
        # For gym environments, use the default buffer (like in EPLHb-Gym.py)
        # This ensures compatibility with seq_len=step training
        print("Warm up: fill the replay buffer with random actions...")
        # Create a separate environment for warm-up with no rendering
        warmup_env = gym.make(env.unwrapped.spec.id, render_mode=None)
        obs, _ = warmup_env.reset()
        for _ in tqdm(range(warmup_steps), desc="Warm-up steps"):
            action = warmup_env.action_space.sample()
            next_obs, reward, terminated, truncated, info = warmup_env.step(int(action))
            done = terminated or truncated
            def to_obs_array(o):
                return o if isinstance(o, np.ndarray) else np.array([o])
            model.replay_buffer.add(
                obs=to_obs_array(obs),
                next_obs=to_obs_array(next_obs),
                action=np.array([action]),
                reward=np.array([reward], dtype=np.float32),
                done=done,
                infos=[info],
                d=np.array([0.0], dtype=np.float32),  # Not used in standard gym
                z=np.array([0.0], dtype=np.float32),  # Not used in standard gym
            )
            obs = next_obs
            if done:
                obs, _ = warmup_env.reset()
        # Close warm-up environment
        warmup_env.close()
        return None  # No custom buffer needed, use default
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_operant_environment(model, env, env_params, pid_params, orig_buffer, fix_source_weights=0):
    """Train on operant environment (exactly matching EPLHb-Operant.py)"""

    print(f"Training on operant environment")
    
    # Setup logging and recorder
    model.set_logger(configure(None, []))
    recorder = SessionRecorder()
    
    # Training parameters
    num_trials = env_params["num_trials"]
    enl_threshold = env_params["enl_threshold"]
    enl_punish_scale = env_params["enl_punish_scale"]
    gradient_steps = pid_params["gradient_steps"]
    
    # Epsilon decay
    eps_start = pid_params["initial_eps"]
    eps_end = pid_params["minimum_eps"]
    max_num_iters = 40000
    decay_trials = int(pid_params["exploration_fraction"] * max_num_iters)
    
    # Start training
    obs, _ = env.reset()
    trial_idx = 0
    eps = eps_start
    recorder._prev_obs = obs
    
    while trial_idx < num_trials:
        # Get reference to the network (always needed)
        q_net = model.policy.q_net
        
        # Manage network freezing for the first n trials
        if trial_idx < fix_source_weights:
            print(f"\n--- Phase 1: Freezing transferred weights for first {fix_source_weights} trials ---")
            # Freeze transferred weights (RNN, MLP body, EPLHb)
            for name, param in q_net.named_parameters():
                if any(layer in name for layer in ['rnn', 'eplhb', 'eplhb_coeff_raw']):
                    param.requires_grad = False
                elif 'post_rnn' in name and not name.endswith('.weight') and not name.endswith('.bias'):
                    # Freeze MLP body layers (all but the last output layer)
                    param.requires_grad = False
        
            # Create optimizer that only manages unfrozen parameters
            model.policy.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, q_net.parameters()), 
                lr=pid_params['learning_rate']
            )
        
        if trial_idx >= fix_source_weights:
            print(f"\n--- Phase 2: Unfreezing all weights to fine-tune entire network ---")
            # Unfreeze all parameters
            for param in q_net.parameters():
                param.requires_grad = True
            
            # Rebuild optimizer to manage all parameters
            model.policy._build(model.lr_schedule)

        print(f"Trial {trial_idx+1}/{num_trials}, ε={eps:.3f}") 
        
        # Reset RNN state
        model.policy.q_net.reset_hidden(batch_size=pid_params["batch_size"])
        done = False
        trial_timesteps = 0
        enl_count = 0
        z_prev = 0.0
        
        # Run one trial
        while not done:
            # Set exploration rate
            model.exploration_rate = eps
            model.logger.record("rollout/exploration_rate", eps)
            
            # Act
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, _, _, info = env.step(action)
            done = info["done"]
            outcome = info["outcome"]
            
            # Punish if stuck in ENL for > threshold steps
            enl_count = enl_count + 1 if outcome and "enl" in outcome else 0
            reward -= max(enl_count - enl_threshold, 0) * enl_punish_scale
            
            # Update gains and sync networks
            model._on_step()
            trial_timesteps += 1
            
            # Calculate d and z updates for replay buffer
            with torch.no_grad():
                # Make observation tensor
                obs_t = torch.tensor(obs, device=model.device, dtype=torch.float32).unsqueeze(0)
                next_t = torch.tensor(next_obs, device=model.device, dtype=torch.float32).unsqueeze(0)
                
                # Get the D update
                if model.tabular_d:
                    d_update = model.gain_adapter.get_d_update(obs_t, next_t)
                else:
                    d_out = model.d_net(obs_t)
                    d_update = d_out[0, action].item()
                
                # Get PID gains
                action_scalar = int(action)
                a_t = torch.tensor([[action_scalar]], dtype=torch.long, device=model.device)
                kp, ki, kd, alpha, beta = model.gain_adapter.get_gains(obs_t, a_t, None)
                
                # Q-values for TD-error
                q_curr = model.policy.q_net(obs_t)[0, action].item()
                q_next = model.policy.q_net_target(next_t).max(dim=1)[0].item()
                td_err = reward + (0.0 if done else model.gamma * q_next) - q_curr
                
                # Integrator update
                z_update = beta * z_prev + alpha * td_err
                
                # Update Q networks for RNN state
                q_cue = model.policy.q_net(obs_t)[0, action_scalar].item()
                q_reward = model.policy.q_net(next_t)[0, action_scalar].item()
            
            # Add to replay buffer
            model.replay_buffer.add(
                obs=obs, next_obs=next_obs,
                action=np.array([action]),
                reward=np.array([reward], dtype=np.float32),
                done=done, infos=[info],
                d=np.array([d_update], dtype=np.float32),
                z=np.array([z_update], dtype=np.float32),
            )
            
            # Record every timestep
            recorder.record_env_step(trial_idx, action, reward, next_obs, info, model=model,
                                  record_sign_index=True, record_eplhb_weight=True)
            
            # Update obs, z_prev
            obs, z_prev = next_obs, z_update
        
        if outcome == "trial_end":
            trial_idx += 1
            # Compute step-based epsilon
            frac = min(1.0, trial_idx / max(1, decay_trials))
            eps = eps_start + frac * (eps_end - eps_start)
        
        # Do training step
        model.train(batch_size=pid_params["batch_size"], seq_len=trial_timesteps, gradient_steps=gradient_steps)
        
        # Restore original buffer to keep accumulating long-term experience
        if orig_buffer is not None:
            model.replay_buffer = orig_buffer
    
    print(f"Operant environment training complete! Trained for {num_trials} trials.")
    return recorder

def train_gym_environment(model, env, env_params, pid_params, fix_source_weights=0):
    """Train on gym environment (exactly matching EPLHb-Gym.py)"""
    
    print(f"Training on gym environment")
    
    # Setup logging
    model.set_logger(configure(None, []))
    
    # Training parameters
    num_episodes = env_params["num_episodes"]
    max_steps = env_params["max_steps"]
    train_every_n_steps = env_params["train_every_n_steps"]
    
    # Epsilon decay
    eps_start = pid_params["initial_eps"]
    eps_end = pid_params["minimum_eps"]
    max_num_iters = num_episodes
    decay_episodes = int(pid_params["exploration_fraction"] * max_num_iters)
    
    # Start training
    eps = eps_start
    all_total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        # Manage network freezing for the first n trials
        if episode < fix_source_weights:
            print(f"\n--- Phase 1: Freezing transferred weights for first {fix_source_weights} episodes ---")
            # Freeze transferred weights (RNN, MLP body, EPLHb)
            q_net = model.policy.q_net
            for name, param in q_net.named_parameters():
                if any(layer in name for layer in ['rnn', 'eplhb', 'eplhb_coeff_raw']):
                    param.requires_grad = False
                elif 'post_rnn' in name and not name.endswith('.weight') and not name.endswith('.bias'):
                    # Freeze MLP body layers (all but the last output layer)
                    param.requires_grad = False
            
            # Create optimizer that only manages unfrozen parameters
            model.policy.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, q_net.parameters()),
                lr=pid_params['learning_rate']
            )
        
        if episode > fix_source_weights:
            print(f"\n--- Phase 2: Unfreezing all weights to fine-tune entire network ---")
            # Unfreeze all parameters
            for param in model.policy.q_net.parameters():
                param.requires_grad = True
            
            # Rebuild optimizer to manage all parameters
            model.policy._build(model.lr_schedule)
        
        while not done and step < max_steps:
            # Set exploration rate
            model.exploration_rate = eps
            model.logger.record("rollout/exploration_rate", eps)
                
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward

                # Update gains and sync networks
            model._on_step()
            
                # Add to replay buffer
            def to_obs_array(o):
                return o if isinstance(o, np.ndarray) else np.array([o])

            model.replay_buffer.add(
                    obs=to_obs_array(obs),
                    next_obs=to_obs_array(next_obs),
                    action=np.array([action]),
                    reward=np.array([reward], dtype=np.float32),
                    done=done,
                    infos=[info],
                    d=np.array([0.0], dtype=np.float32),  # Not used in standard gym
                    z=np.array([0.0], dtype=np.float32),  # Not used in standard gym
                )
            
            obs = next_obs
            step += 1

            # Train every n steps if buffer is sufficiently filled
            if step % train_every_n_steps == 0 and model.replay_buffer.size() > pid_params["learning_starts"]:
                model.train(batch_size=pid_params["batch_size"], seq_len=step, gradient_steps=pid_params["gradient_steps"])

        # Store per-episode rewards
        all_total_rewards.append(total_reward)
        
        # Compute step-based epsilon
        frac = min(1.0, episode / max(1, decay_episodes))
        eps = eps_start + frac * (eps_end - eps_start)
        
        print(f"Episode {episode+1}/{num_episodes}: Total Reward: {total_reward}, eps: {eps:.2f}")
    
    print(f"Gym environment training complete! Trained for {num_episodes} episodes.")
    return all_total_rewards

# ============================================================================
# Weight transfer functions
# ============================================================================

def transfer_weights(source_model, target_model, use_projection=True):
    """Transfer compatible weights from source model to target model with optional projection layer"""

    print("Transferring compatible network weights...")

    # Transfer compatible weights from source model to target model
    old_qnet = source_model.policy.q_net
    new_qnet = target_model.policy.q_net
    
    with torch.no_grad():
        # Transfer input projection layer weights if both models have them
        if hasattr(old_qnet, 'input_projection') and hasattr(new_qnet, 'input_projection'):
            try:
                # Check if input dimensions are compatible
                old_input_size = old_qnet.input_projection.in_features
                new_input_size = new_qnet.input_projection.in_features
                
                if old_input_size == new_input_size:
                    # Same input size, transfer directly
                    new_qnet.input_projection.load_state_dict(
                        old_qnet.input_projection.state_dict()
                    )
                    print("✓ Input projection layer weights transferred successfully")
                    
            except Exception as e:
                print(f"⚠ Warning: Input projection layer transfer failed: {e}")
        
        # Transfer input normalization layer weights if both models have them
        if hasattr(old_qnet, 'input_norm') and hasattr(new_qnet, 'input_norm'):
            try:
                # Check if normalization dimensions are compatible
                old_norm_size = old_qnet.input_norm.normalized_shape[0]
                new_norm_size = new_qnet.input_norm.normalized_shape[0]
                
                if old_norm_size == new_norm_size:
                    new_qnet.input_norm.load_state_dict(
                        old_qnet.input_norm.state_dict()
                    )
                    print("✓ Input normalization layer weights transferred successfully")
                    
            except Exception as e:
                print(f"⚠ Warning: Input normalization layer transfer failed: {e}")
        
        # Transfer EPLHb input normalization layer weights if both models have them
        if hasattr(old_qnet, 'eplhb_input_norm') and hasattr(new_qnet, 'eplhb_input_norm'):
            try:
                # Check if normalization dimensions are compatible
                old_norm_size = old_qnet.eplhb_input_norm.normalized_shape[0]
                new_norm_size = new_qnet.eplhb_input_norm.normalized_shape[0]
                
                if old_norm_size == new_norm_size:
                    new_qnet.eplhb_input_norm.load_state_dict(
                        old_qnet.eplhb_input_norm.state_dict()
                    )
                    print("✓ EPLHb input normalization layer weights transferred successfully")
                else:
                    print(f"⚠ EPLHb normalization size mismatch: {old_norm_size} -> {new_norm_size}")
                    print("   Creating new EPLHb input normalization layer for target model...")
                    
                    # Create a new normalization layer with the correct dimensions
                    new_norm = nn.LayerNorm(new_norm_size)
                    new_qnet.eplhb_input_norm = new_norm
                    print("✓ Created new EPLHb input normalization layer for target model")
                    
            except Exception as e:
                print(f"⚠ Warning: EPLHb input normalization layer transfer failed: {e}")
        
        # Transfer RNN weights if present
        if hasattr(old_qnet, 'rnn') and hasattr(new_qnet, 'rnn'):
            try:
                # Now that we have input projection layers, we can transfer RNN weights
                # The input projection will handle the observation space conversion
                new_qnet.rnn.load_state_dict(old_qnet.rnn.state_dict())
                print("✓ RNN weights transferred successfully (with input projection)")
                    
            except Exception as e:
                print(f"⚠ Warning: RNN weight transfer failed: {e}")
        
        # Transfer post_rnn (MLP head) except for the last layer (output head)
        if hasattr(old_qnet, 'post_rnn') and hasattr(new_qnet, 'post_rnn'):
            old_layers = list(old_qnet.post_rnn.children())
            new_layers = list(new_qnet.post_rnn.children())
            min_len = min(len(old_layers), len(new_layers))
            
            for i in range(min_len-1):  # Skip last layer (output head)
                try:
                    new_layers[i].load_state_dict(old_layers[i].state_dict())
                    print(f"✓ MLP layer {i} weights transferred successfully")
                except Exception as e:
                    print(f"⚠ Warning: MLP layer {i} transfer failed: {e}")
        
        # Transfer EPLHb layer if present
        if hasattr(old_qnet, 'eplhb') and hasattr(new_qnet, 'eplhb'):
            try:
                new_qnet.eplhb.load_state_dict(old_qnet.eplhb.state_dict())
                print("✓ EPLHb weights transferred successfully")
            except Exception as e:
                print(f"⚠ Warning: EPLHb weight transfer failed: {e}")
        
        # Transfer eplhb_coeff_raw if present
        if hasattr(old_qnet, 'eplhb_coeff_raw') and hasattr(new_qnet, 'eplhb_coeff_raw'):
            new_qnet.eplhb_coeff_raw.data.copy_(old_qnet.eplhb_coeff_raw.data)
            print("✓ EPLHb coefficient transferred successfully")


def save_and_plot_results(source_recorder, all_total_rewards, source_env_type, target_env_type, source_env_params,
                        save=True, plot=True):
    """Save results and generate plots"""
    print(f"\n{'='*60}")
    print("Saving results and plotting")
    print(f"{'='*60}")
    
    if save:
        # Save the recorder and reward data with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save source environment results
        if source_recorder:
            import pickle
            source_filename = f"{timestamp}-{source_env_type}_results.pkl"
            # Store both params and recorder
            results = {
                "session_params": source_env_params,
                "pid_params":     operant_pid_params,
                "recorder":       source_recorder,
                "seed":           operant_pid_params["seed"],
            }

            # Save everything
            with open(f"PID-results/{timestamp}-{source_env_type}_results.pkl", "wb") as f:
                pickle.dump(results, f)
            print(f"Saved source environment results to {source_filename}")
        
        # Save target environment results
        if all_total_rewards is not None:
            # Save as pickle file
            results = {
                "session_params": target_env_params,
                "pid_params":     target_pid_params,
                "recorder":       None,
                "seed":           target_pid_params["seed"],
            }
            with open(f"PID-results/{timestamp}-{target_env_type}_results.pkl", "wb") as f:
                pickle.dump(results, f)
            print(f"Saved target environment results to {timestamp}-{target_env_type}_results.pkl")
    
    if plot:
        # Plot source environment results
        if source_env_type == 'operant' and source_recorder:
            print("\n--- Plotting results from Operant Task ---")
            plot_figure(source_recorder, td_error_type='internal', dt=source_env_params["dt"], show=True,
                    pre_steps=source_env_params["pre_steps"], post_steps=source_env_params["post_steps"])
        
        # Plot target environment results
        if all_total_rewards is not None:
            print("\n--- Plotting results from Target Environment ---")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(all_total_rewards, label='Total Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'{target_env_type.title()} Transfer Learning Performance')
            plt.legend()
            plt.grid(True)
            plt.show()

# ============================================================================
# MAIN TRANSFER LEARNING FUNCTION
# ============================================================================
def run_transfer_learning():
    """Main function to run transfer learning between environments"""
    print("=" * 60)
    print("EPLHb Transfer Learning")
    print("=" * 60)
    print(f"Source Environment: {transfer_params['source_env']}")
    print(f"Target Environment: {transfer_params['target_env']}")
    print("=" * 60)
    
    # Setup source environment and model
    print(f"\nSetting up source environment: {transfer_params['source_env']}")
    source_env, source_env_params, source_pid_params = setup_environment(transfer_params['source_env'])
    source_model, source_gain_adapter = setup_model(source_env, source_pid_params)
    
    
    # Setup target environment and model
    print(f"\nSetting up target environment: {transfer_params['target_env']}")
    target_env, target_env_params, target_pid_params = setup_environment(transfer_params['target_env'])
    target_model, target_gain_adapter = setup_model(target_env, target_pid_params)
    
    print("\nTransfer learning setup complete!")
    
    # ============================================================================
    # PHASE 1: TRAIN ON SOURCE ENVIRONMENT
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training on {transfer_params['source_env']} environment")
    print(f"{'='*60}")
    
    source_orig_buffer = setup_buffer(source_model, transfer_params['source_env'], source_env)
    if transfer_params['source_env'] == 'operant':
        source_recorder = train_operant_environment(source_model, source_env, source_env_params, source_pid_params, source_orig_buffer)
    else:
        source_recorder = None
        print("Source environment training skipped (not operant)")

    # Plot results from source environment
    save_and_plot_results(source_recorder, None, transfer_params['source_env'], transfer_params['target_env'], source_env_params, save=False, plot=True)
    
    # ============================================================================
    # PHASE 2: TRANSFER WEIGHTS TO TARGET MODEL
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: Transferring weights to {transfer_params['target_env']} model")
    print(f"{'='*60}")
    
        # Use projection layer for weight transfer to handle different observation space sizes
    transfer_weights(source_model, target_model, use_projection=True)
    
    # ============================================================================
    # PHASE 3: TRAIN ON TARGET ENVIRONMENT
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 3: Training on {transfer_params['target_env']} environment")
    print(f"{'='*60}")
    
    target_orig_buffer = setup_buffer(target_model, transfer_params['target_env'], target_env)    
    if transfer_params['target_env'] == 'gym':
        all_total_rewards = train_gym_environment(
            target_model, target_env, target_env_params, target_pid_params, transfer_params['fix_source_weights']
        )
    else:
        all_total_rewards = None
        print("Target environment training skipped (not gym)")

    # Plot results from target environment
    save_and_plot_results(None, all_total_rewards, transfer_params['target_env'], transfer_params['target_env'], target_env_params, save=False, plot=True)
    
    # ============================================================================
    # PHASE 4: SAVE RESULTS AND PLOT
    # ============================================================================
    save_and_plot_results(source_recorder, all_total_rewards, 
                         transfer_params['source_env'], transfer_params['target_env'], 
                         source_env_params)
    
    print("\n🎉 Transfer learning complete!")
    print(f"Successfully transferred from {transfer_params['source_env']} to {transfer_params['target_env']}")
    if transfer_params['fix_source_weights'] > 0:
        print(f"Fixed transferred weights for first {transfer_params['fix_source_weights']} episodes")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_transfer_learning()

