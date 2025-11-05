#!/usr/bin/env python3
import os, sys, importlib, pickle, itertools, random
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer
from TabularPID.Agents.DQN.DQN import EPLHb_DQN, PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

from OperantGym import OperantLearning
from plotfunctions import plot_figure
from summary_plots import plot_pid_results
from recorder import SessionRecorder
from trainfuntions import set_global_seeds, setup_model, setup_buffer, train_operant_environment, train_gym_environment, transfer_weights


# ----------------------------
# Experiment sweep parameters
# ----------------------------
experiment_params = {
    "kd_values":        [0],   # sweep over kd (add values as needed)
    "omission_probs":   [0.3],   # sweep over omission probability
    "repeats":          10,     # repeats per combination
    "max_batch_sizes":  [1],   # grid values for max_batch_size
    "num_recents":      [1],   # grid values for num_recent
    "eplhb_fixed_sign": [True, False]
}
seed = 23


# ============================================================================
# OPERANT ENVIRONMENT PARAMETERS (exactly matching EPLHb-Operant.py)
# ============================================================================
operant_session_params = {
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

    "dt":               0.1,          # 100 ms

    "continual_learning": False,
    "change_start": 200,
    "change_interval": 50,
}

operant_pid_params = {
    "learning_rate": 1e-3,
    "eplhb_lr": 1e-2,
    "coeff_lr": 0.0,
    "initial_eplhb_coeff": -0.3,
    "rnn_type": "GRU",
    "l2_lambda": 0.0,
    "batch_training": False,
    "batch_size": 1,
    "buffer_size": 1,
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
    "fixed_sign": True,
    "eplhb_fixed_sign": False,
}

# ============================================================================
# GYM ENVIRONMENT PARAMETERS (exactly matching EPLHb-Gym.py)
# ============================================================================
gym_env_params = {
    "env_name": "CliffWalking-v0",  # Change to "CliffWalking-v0" or any Gymnasium env
    "num_episodes": 200,
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
            continual_learning=operant_session_params["continual_learning"],
            change_start=operant_session_params["change_start"],
            change_interval=operant_session_params["change_interval"],
            print_status=False,
        )
        return env, operant_session_params, operant_pid_params
    elif env_type == "gym":
        env = gym.make(gym_env_params["env_name"], render_mode=gym_env_params["render_mode"])
        return env, gym_env_params, gym_pid_params
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

# ============================================================================
# Weight transfer functions
# ============================================================================
def save_and_plot_results(env_type, env_params, pid_params, results=None,
                        recorder=None, stuck_counts=None, save_name=None, reward_history=None, 
                        save=True, plot=True, r=None, kd=None, omit=None, max_b=None, num_r=None, fixed_sign=None, eplhb_fixed_sign=None):
    """Save results and generate plots"""
    print(f"\n{'='*60}")
    print("Saving results and plotting")
    print(f"{'='*60}")

    if results is None:
        results = {}
    
    if save:
        # Store both params and recorder
        results[(kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign, r)] = {
            "session_params": env_params,
            "pid_params":     env_params,
            "recorder":       recorder,
            "seed":           pid_params["seed"],
            "stuck_counts":   stuck_counts
        }
    
    if plot:
        # Plot source environment results
        if env_type == 'operant' and recorder:
            print("\n--- Plotting results from Operant Task ---")
            plot_figure(recorder, dt=env_params["dt"], pre_steps=env_params["pre_steps"], post_steps=env_params["post_steps"],
                        save=save, save_path=os.path.join(save_dir, save_name),show=False)
    
    else:
        results = None
    
    return results

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
    
    # Setup target environment and model
    print(f"\nSetting up target environment: {transfer_params['target_env']}")
    target_env, target_env_params, target_pid_params = setup_environment(transfer_params['target_env'])

    # define results
    global results
    
    print("\nTransfer learning setup complete!")
    
    # ============================================================================
    # PHASE 1: TRAIN ON SOURCE ENVIRONMENT
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training on {transfer_params['source_env']} environment")
    print(f"{'='*60}")
    
    # We will create models per run; buffer creation is handled inside training
    if transfer_params['source_env'] == 'operant':
        # Define the three network types
        network_types = [
            {"name": "ANN", "fixed_sign": False, "eplhb_fixed_sign": False},
            {"name": "EPLHb", "fixed_sign": True,  "eplhb_fixed_sign": False},
            {"name": "Dales", "fixed_sign": True,  "eplhb_fixed_sign": True},
        ]

        for r in range(experiment_params["repeats"]):
            # Track stuck counts per network type across retries for this repeat
            per_type_stuck_counts = {nt["name"]: 0 for nt in network_types}

            while True:
                # Use the same seed for all three networks in this attempt
                new_seed = random.randint(0, 10000)
                set_global_seeds(new_seed)

                # Run all network types with the same seed
                results_this_attempt = {}
                any_stuck = False
                for nt in network_types:
                    # Update params for this network type
                    source_pid_params["seed"] = new_seed
                    source_pid_params["fixed_sign"] = nt["fixed_sign"]
                    source_pid_params["eplhb_fixed_sign"] = nt["eplhb_fixed_sign"]

                    print(
                        f"Training with fixed_sign={nt['fixed_sign']}, eplhb_fixed_sign={nt['eplhb_fixed_sign']}, "
                        f"kd={kd}, omit={omit}, max_batch={max_b}, num_recent={num_r}, "
                        f"(repeat {r + 1}/{repeats})"
                    )

                    # Fresh model per run
                    source_model, _ = setup_model(source_env, source_pid_params, model_type="EPLHb")
                    source_recorder, retrain, got_stuck = train_operant_environment(
                        source_model, source_env, source_env_params, source_pid_params, print_status=False, save_dir=save_dir
                    )

                    if got_stuck:
                        per_type_stuck_counts[nt["name"]] += 1
                        any_stuck = True

                    # Store for potential saving if this attempt succeeds
                    results_this_attempt[nt["name"]] = {
                        "recorder": source_recorder,
                        "fixed_sign": nt["fixed_sign"],
                        "eplhb_fixed_sign": nt["eplhb_fixed_sign"],
                    }

                # If any network got stuck with this seed, retry with a new seed
                if any_stuck:
                    print("One or more networks got stuck; retrying with a new seed...")
                    continue

                # Success: all three networks ran without getting stuck
                for nt in network_types:
                    save_name = (
                        f"kd_{kd}_omit_{omit}_maxB_{max_b}_numR_{num_r}_"
                        f"fixedSign_{nt['fixed_sign']}_eplhbfixedSign_{nt['eplhb_fixed_sign']}_seed_{new_seed}.png"
                    )

                    results = save_and_plot_results(
                        transfer_params['source_env'],
                        source_env_params,
                        source_pid_params,
                        results=results,
                        recorder=results_this_attempt[nt["name"]]["recorder"],
                        stuck_counts=per_type_stuck_counts[nt["name"]],
                        save_name=save_name,
                        r=r,
                        kd=kd, omit=omit, max_b=max_b, num_r=num_r,
                        fixed_sign=nt['fixed_sign'], eplhb_fixed_sign=nt['eplhb_fixed_sign']
                    )

                # Exit the retry loop for this repeat
                break

        # Save everything per (kd, omit, max_b, num_r) combination
        result_file = (
            f"results_Kd_{kd}_omit_{omit}_maxB_{max_b}_numR_{num_r}.pkl"
        )
        with open(os.path.join(save_dir, result_file), "wb") as f:
            pickle.dump(results, f)

    else:
        source_recorder = None
        print("Source environment training skipped (not operant)")

    
    # # ============================================================================
    # # PHASE 2: TRANSFER WEIGHTS TO TARGET MODEL
    # # ============================================================================
    # print(f"\n{'='*60}")
    # print(f"PHASE 2: Transferring weights to {transfer_params['target_env']} model")
    # print(f"{'='*60}")
    
    # # Setup target model
    # target_model, _ = setup_model(target_env, target_pid_params)

    # # Use projection layer for weight transfer to handle different observation space sizes
    # transfer_weights(source_model, target_model, use_projection=True)
    
    # # ============================================================================
    # # PHASE 3: TRAIN ON TARGET ENVIRONMENT
    # # ============================================================================
    # print(f"\n{'='*60}")
    # print(f"PHASE 3: Training on {transfer_params['target_env']} environment")
    # print(f"{'='*60}")
    
    # _ = setup_buffer(target_model, transfer_params['target_env'], target_env)    
    # if transfer_params['target_env'] == 'gym':
    #     all_total_rewards = train_gym_environment(
    #         target_model, target_env, target_env_params, target_pid_params, transfer_params['fix_source_weights']
    #     )
    # else:
    #     all_total_rewards = None
    #     print("Target environment training skipped (not gym)")

    # # Plot results from target environment
    # save_and_plot_results(transfer_params['target_env'], target_env_params, target_pid_params, reward_history=all_total_rewards, save=False, plot=True)
    
    # # ============================================================================
    # # PHASE 4: SAVE RESULTS AND PLOT
    # # ============================================================================
    # # Save source environment results
    # save_and_plot_results(transfer_params['source_env'], source_env_params, source_pid_params, recorder=source_recorder, save=True, plot=False)
    # # Save target environment results
    # save_and_plot_results(transfer_params['target_env'], target_env_params, target_pid_params, reward_history=all_total_rewards, save=True, plot=False)
    
    # print("\n🎉 Transfer learning complete!")
    # print(f"Successfully transferred from {transfer_params['source_env']} to {transfer_params['target_env']}")
    # if transfer_params['fix_source_weights'] > 0:
    #     print(f"Fixed transferred weights for first {transfer_params['fix_source_weights']} episodes")

# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    # Define sweep grid
    kd_values        = experiment_params["kd_values"]
    omission_probs   = experiment_params["omission_probs"]
    repeats          = experiment_params["repeats"]

    max_batch_sizes  = experiment_params["max_batch_sizes"]
    num_recents      = experiment_params["num_recents"]
    eplhb_fixed_sign = experiment_params["eplhb_fixed_sign"]

    # Save results settings
    batch_name = 'kd_omission_sweep'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(script_dir, "PID-results-ext_buffer")

    os.makedirs(results_root, exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y%m%d")
    save_dir = os.path.join(results_root, f"{today}-{batch_name}")
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # --- New sweep loop ---
    for max_b, num_r in zip(max_batch_sizes, num_recents):
        print(f"\n=== Testing max_batch_size={max_b}, num_recent={num_r} ===")

        for kd, omit in itertools.product(
            kd_values, omission_probs
        ):
            operant_session_params["omission_prob"] = omit
            operant_session_params["max_batch_size"] = max_b
            operant_session_params["num_recent"] = num_r
            operant_pid_params["kd"] = kd
            operant_pid_params["meta_lr_d"] = min(kd, 0.1)
            # eplhb/fixed_sign variants handled inside run_transfer_learning

            print(f"\n--- Running sweep: kd={kd}, omission_prob={omit}, "
                f"max_batch_size={max_b}, buffer_trials_recent={num_r} ---")

            run_transfer_learning()

    plot_pid_results(results_root)

