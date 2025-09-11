import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer, ExtendedReplayBuffer
from TabularPID.Agents.DQN.DQN import EPLHb_DQN, PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import random
from recorder import SessionRecorder

# TODO: Add PID model setup function
#def setup_PID_model(env, pid_params, device="cpu"):



def setup_EPLHb_model(env, pid_params, device="cpu"):
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
    if device == "cpu":
        import TabularPID.Agents.DQN.DQN_policy as _dp
        for _name in dir(_dp):
            cls = getattr(_dp, _name)
            if isinstance(cls, type) and hasattr(cls, "jump_start_cuda"):
                cls.jump_start_cuda = lambda self: None
    else:
        device = torch.device(device)

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

        replay_buffer_class=ExtendedReplayBuffer,
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
        model.replay_buffer = ExtendedReplayBuffer(
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



def train_operant_environment(model, env, env_params, pid_params, orig_buffer,
                              fix_source_weights=0,
                              change_start=0,
                              change_interval=0,
                              pairing_change=False,
                              difficulty_change="increase",
                              print_status=True,
                              seq_len=10):
    """Train on operant environment (matching PID-Operant-Batch.py structure)"""

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
    enl_count = 0
    eps = eps_start
    recorder._prev_obs = obs
    final_indices = []
    
    # Phase printing flags
    phase1_printed = False
    phase2_printed = False
    
    # --- Main training loop with tqdm bar ---
    pbar = tqdm(total=num_trials,
                desc=f"Trials (seed={pid_params['seed']})",
                unit="trial")
    retrain = False
    
    while trial_idx < num_trials:
        # Get reference to the network (always needed)
        q_net = model.policy.q_net
        
        # Manage network freezing for the first n trials
        if fix_source_weights > 0:  # Only do weight freezing if actually needed
            if trial_idx < fix_source_weights:
                if not phase1_printed:
                    print(f"\n--- Phase 1: Freezing transferred weights for first {fix_source_weights} trials ---")
                    phase1_printed = True
                    # Freeze transferred weights (RNN, MLP body, EPLHb)
                    for name, param in q_net.named_parameters():
                        if any(layer in name for layer in ['rnn', 'eplhb', 'eplhb_coeff_raw']):
                            param.requires_grad = False
                        elif 'post_rnn' in name and not name.endswith('.weight') and not name.endswith('.bias'):
                            # Freeze MLP body layers (all but the last output layer)
                            param.requires_grad = False
                    
                    # Create optimizer that only manages unfrozen parameters with correct learning rates
                    rebuild_optimizer_with_correct_lr_groups(model, pid_params)
            
            elif trial_idx == fix_source_weights:  # Only transition once
                if not phase2_printed:
                    print(f"\n--- Phase 2: Unfreezing all weights to fine-tune entire network ---")
                    phase2_printed = True
                    # Unfreeze all parameters
                    for param in q_net.parameters():
                        param.requires_grad = True
                    # Rebuild optimizer with correct learning rate groups
                    rebuild_optimizer_with_correct_lr_groups(model, pid_params)

        if print_status:
            print(f"Trial {trial_idx+1}/{num_trials}, ε={eps:.3f}") 
        
        # Reset RNN state
        model.policy.q_net.reset_hidden(batch_size=pid_params["batch_size"])
        done = False
        trial_timesteps = 0
        trial_inds = [] # Reset trial indices for this trial
        z_prev = 0.0
        env.trial_count = trial_idx

        # Make changes for continual learning
        # if change_interval > 0 and trial_idx >= change_start and (trial_idx - change_start) % change_interval == 0:
        #     if pairing_change:
        #         # Change pairing type randomly
        #         env_params["pairing"] = random.choice(['reward', 'punish'])
        #         print(f"Changing pairing to {env_params['pairing']} at trial {trial_idx}")
        #     else:
        #         if difficulty_change == "increase":
        #             env_params["omission_prob"] = min(0.1, env_params["omission_prob"] + 0.1)
        #         elif difficulty_change == "decrease":
        #             env_params["omission_prob"] = max(0.0, env_params["omission_prob"] - 0.1)
        #         elif difficulty_change == "random":
        #             env_params["omission_prob"] = random.choice(np.arange(0.0, 0.9, 0.1))
        #         elif difficulty_change == "bandit":
        #             # make omission_prob switch from 0.2 to 0.8 or vice versa every 10 trials
        #             if trial_idx % 10 == 0 and env_params["omission_prob"] == 0.2:
        #                 env_params["omission_prob"] = 0.8
        #             elif trial_idx % 10 == 0 and env_params["omission_prob"] == 0.8:
        #                 env_params["omission_prob"] = 0.2

        # run one trial
        while not done:
            # set exploration rate
            model.exploration_rate = eps
            model.logger.record("rollout/exploration_rate", eps)

            # act
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, _, _, info = env.step(action)
            done = info["done"]
            outcome = info["outcome"]
            
            # Punish if stuck in ENL for > threshold steps
            # enl_count = enl_count + 1 if outcome and "enl" in outcome else 0
            # reward -= max(enl_count - enl_threshold, 0) * enl_punish_scale
            
            # Update gains and sync networks
            model._on_step()
            trial_timesteps += 1

            # calcualte d and z updates for the replay buffer
            with torch.no_grad():
                # make observation tensor
                obs_t  = torch.tensor(obs,  device=model.device, dtype=torch.float32).unsqueeze(0)
                next_t = torch.tensor(next_obs, device=model.device, dtype=torch.float32).unsqueeze(0)

                # get d update
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
            
            # Record current trial idx within buffer (like in PID-Operant.py)
            idx = (model.replay_buffer.pos - 1) % model.replay_buffer.buffer_size
            trial_inds.append(idx)
            
            # Record every timestep
            recorder.record_env_step(trial_idx, action, reward, next_obs, info, model=model,
                                  record_sign_index=True, record_eplhb_weight=True)
            
            # Update obs, z_prev
            obs, z_prev = next_obs, z_update

        # update exploration rate upon trial completion
        if outcome == "trial_end":
            trial_idx += 1
            pbar.update(1)
            enl_count = 0
            # Compute step-based epsilon
            frac = min(1.0, trial_idx / max(1, decay_trials))
            eps = eps_start + frac * (eps_end - eps_start)
        else:
            # punish if stuck in ENL for > 200 steps
            enl_count += 1
            reward -= max(enl_count - enl_threshold, 0) * enl_punish_scale
            # reset the seed and retrain if ENL > 1000 steps
            if enl_count > 500:
                retrain = True
                print(f"ENL break after {enl_count} steps, retraining with different seed...")
                return recorder, retrain, True
        
        # Bootstrapping: collect final step indices (like in PID-Operant-Batch.py)
        final_indices.append(trial_inds[-1])
        
        # Make dynamic to adjust for batch size vs. available trials to pull from
        # Use max_batch_size and num_recent from env_params (like PID-Operant-Batch.py)
        max_batch_size = pid_params.get("max_batch_size", 10)
        num_recent = pid_params.get("num_recent", 5)
        
        k = min(len(final_indices), max_batch_size)
        n_recent = min(len(final_indices), num_recent)
        recent = final_indices[-n_recent:]
        remaining = final_indices[:-n_recent]
        needed = k - len(recent)
        if needed > 0 and remaining:
            sampled = random.sample(remaining, min(needed, len(remaining)))
        else:
            sampled = []
        batch_idxs = recent + sampled
        
        if print_status and trial_idx % 10 == 0:  # Print every 10 trials to avoid spam
            print(f"Trial {trial_idx}: batch_idxs={batch_idxs} (k={k}, recent={len(recent)}, sampled={len(sampled)})")
        
        # Do training step using batch_idxs with BPTT
        model.train(gradient_steps=gradient_steps, batch_idxs=batch_idxs)
        
        # Restore original buffer to keep accumulating long-term experience
        # if orig_buffer is not None:
        #     model.replay_buffer = orig_buffer
    
    print(f"Operant environment training complete! Trained for {num_trials} trials.")
    pbar.close()
    return recorder, retrain, False



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



def rebuild_optimizer_with_correct_lr_groups(model, pid_params):
    """
    Properly rebuild the optimizer with correct learning rate groups for EPLHb models.
    This preserves the carefully tuned learning rates for different parameter groups.
    """
    from TabularPID.Agents.DQN.DQN_policy import EPLHbNetwork
    
    # Get the current learning rate schedule
    main_lr = model.lr_schedule(1)
    
    # Extract the original learning rates from optimizer_kwargs
    eplhb_lr = pid_params.get('eplhb_lr', 1e-4)
    coeff_lr = pid_params.get('coeff_lr', 1e-5)
    
    # Get the optimizer class and kwargs
    optimizer_class = model.policy.optimizer_class
    optimizer_kwargs = model.policy.optimizer_kwargs.copy()
    
    # Separate parameters into groups
    q_net = model.policy.q_net
    if isinstance(q_net, EPLHbNetwork):
        eplhb_params = list(q_net.eplhb.parameters())
        # Only include eplhb_coeff_raw if coeff_lr > 0
        if coeff_lr > 0:
            eplhb_coeff_param = [q_net.eplhb_coeff_raw]
        else:
            eplhb_coeff_param = []
    else:
        eplhb_params = []
        eplhb_coeff_param = []
    
    other_params = [
        p for n, p in q_net.named_parameters()
        if not n.startswith('eplhb.') and n != 'eplhb_coeff_raw'
    ]
    
    # Create parameter groups with correct learning rates
    param_groups = [
        {'params': other_params, 'lr': main_lr},
        {'params': eplhb_params, 'lr': eplhb_lr},
    ]
    
    # Only add eplhb_coeff_param group if coeff_lr > 0
    if coeff_lr > 0:
        param_groups.append({'params': eplhb_coeff_param, 'lr': coeff_lr})
    
    # Create new optimizer with correct learning rate groups
    model.policy.optimizer = optimizer_class(
        param_groups,
        **optimizer_kwargs,
    )
    
    print(f"✓ Rebuilt optimizer with learning rates: main={main_lr:.2e}, eplhb={eplhb_lr:.2e}, coeff={coeff_lr:.2e}")


def set_global_seeds(seed: int):
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (may slow you down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False