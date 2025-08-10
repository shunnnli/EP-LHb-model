#!/usr/bin/env python3
import os, sys, copy, pickle, itertools, random
import pandas as pd

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
# from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN # not working for me
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer
from TabularPID.Agents.DQN.DQN import PID_DQN, EPLHb_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

from OperantGym import OperantLearning
from plotfunctions import plot_figure
from recorder import SessionRecorder
from types import SimpleNamespace

# Base hyperparameters from your original script
# session_params defined here
base_session_params = {
    "pairing":          'reward',
    "num_trials":       200,
    "pre_steps":        10,
    "post_steps":       40,
    "enl_duration":     (2.0, 4.0),
    "tau_on":           0.01,
    "tau_off":          0.1,
    "omission_prob":    0.2,
    "action_cost":      0.1,
    "enl_penalty":      0.2,
    "enl_threshold":    200,
    "enl_punish_scale": 0.5,
    "gradient_steps":   10,
    "gamma":            0.95,
    "batch_training":   False,
    "batch_size":       1,
    "buffer_size":      1,
    "dt":               0.1,

    # continual learning settings
    "change_start": 100,  # start changing parameters after this many trials
    "change_interval": 30,
    "difficulty_change": "random", # 'none', 'increase', 'decrease', 'random'
    "pairing_change": False,
}

# pid_params defined here
base_pid_params = {
    "kp":                   1.0,
    "ki":                   0.0,
    "kd":                   0.3,
    "eplhb_lr":             1e-2,
    "coeff_lr":             0.0,
    "initial_eplhb_coeff":  -1,
    "meta_lr":              0,
    "epsilon_gain":         0.1,
    "alpha":                0.05,
    "beta":                 0.95,
    "d_tau":                1,
    "tabular_d":            False,
    "learning_rate":        1e-3,
    "replay_memory_size":   base_session_params["buffer_size"],
    "batch_size":           base_session_params["batch_size"],
    "tau":                  1,
    "gamma":                base_session_params["gamma"],
    "gradient_steps":       1,
    "train_freq":           1,
    "target_update_interval": 10,
    "initial_eps":          0.1,
    "exploration_fraction": 0.001,
    "minimum_eps":          0.05,
    "learning_starts":      1000,
    "inner_size":           64,
    "dump_buffer":          False,
    "is_double":            False,
    "policy_evaluation":    False,
    "seed":                 26,

    "rnn_type":             "GRU",  # Options: "RNN", "GRU", "LSTM". Change as needed.
    "l2_lambda":            1e-4,  # L2 regularization strength for EPLHb weights
}


# --------------------------------------------------------
# Functions
# --------------------------------------------------------
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
    # PID seed
    base_pid_params["seed"] = seed
    pp["seed"] = seed

def train_once(session_params, pid_params):
    # --- Build env, model, recorder ---
    env = OperantLearning(
        pairing=session_params["pairing"],
        omission_prob=session_params["omission_prob"],
        enl_duration=session_params["enl_duration"],
        action_cost=session_params["action_cost"],
        enl_penalty=session_params["enl_penalty"],
        print_status=False
    )

    gain_adapter = SingleGainAdapter(
        kp=pid_params["kp"],
        ki=pid_params["ki"],
        kd=pid_params["kd"],
        alpha=pid_params["alpha"],
        beta=pid_params["beta"],
        meta_lr=pid_params["meta_lr"],
        epsilon=pid_params["epsilon_gain"],
    )

    policy_kwargs = dict(
        net_arch=[pid_params["inner_size"], pid_params["inner_size"]],
        optimizer_class=optim.Adam,
        with_RNN_layer=True,
        rnn_type=pid_params["rnn_type"],  # Options: "RNN", "GRU", "LSTM". Change as needed.
    )

    model = EPLHb_DQN(
        pid_params["d_tau"],
        pid_params["tabular_d"],
        gain_adapter,
        policy="MlpPolicy",
        env=env,
        learning_rate=pid_params["learning_rate"],
        buffer_size=pid_params["replay_memory_size"],
        batch_size=pid_params["batch_size"],
        tau=pid_params["tau"],
        gamma=pid_params["gamma"],
        gradient_steps=pid_params["gradient_steps"],
        train_freq=pid_params["train_freq"],
        target_update_interval=pid_params["target_update_interval"],
        exploration_fraction=pid_params["exploration_fraction"],
        exploration_initial_eps=pid_params["initial_eps"],
        exploration_final_eps=pid_params["minimum_eps"],
        learning_starts=pid_params["learning_starts"],
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        seed=pid_params["seed"],
        device="cpu",
        dump_buffer=pid_params["dump_buffer"],
        is_double=pid_params["is_double"],
        optimal_model=None,
        policy_evaluation=pid_params["policy_evaluation"],
        replay_buffer_class=OnlineReplayBuffer,
    )
    gain_adapter.set_model(model)

    # Set up logging
    # SB3 will no longer spam stdout
    model.set_logger(configure(None, []))
    recorder = SessionRecorder()

    # epsilon decay params
    max_num_iters = 40000
    decay_trials = int(pid_params["exploration_fraction"] * max_num_iters)

    # Replace buffer for on-trial data
    orig_buffer = model.replay_buffer
    model.replay_buffer = OnlineReplayBuffer(
        buffer_size=10_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    # --- Main training loop with tqdm bar ---
    num_trials = session_params["num_trials"]
    pbar = tqdm(total=num_trials,
                desc=f"Trials (kd={pid_params['kd']}, omit={session_params['omission_prob']}, seed={pid_params['seed']})",
                unit="trial")
    retrain = False

    # Set continual learning flag
    change_start = session_params["change_start"]
    change_interval = session_params["change_interval"]
    pairing_change = session_params["pairing_change"]
    difficulty_change = session_params["difficulty_change"]


    obs, _ = env.reset()
    trial_idx = 0
    eps = pid_params["initial_eps"]
    enl_count = 0
    # — prime the recorder so rec._prev_obs isn't None on step 0 —
    recorder._prev_obs = obs

    while trial_idx < num_trials:
        # reset RNN state
        model.policy.q_net.reset_hidden(batch_size=session_params["batch_size"])
        done = False
        trial_timesteps = 0
        z_prev = 0.0

        # Make changes for continual learning
        if trial_idx >= change_start and (trial_idx - change_start) % change_interval == 0:
            if pairing_change:
                # Change pairing type randomly
                session_params["pairing"] = random.choice(['reward', 'punish'])
                print(f"Changing pairing to {session_params['pairing']} at trial {trial_idx}")
            else:
                if difficulty_change == "increase":
                    session_params["omission_prob"] = min(0.1, session_params["omission_prob"] + 0.1)
                elif difficulty_change == "decrease":
                    session_params["omission_prob"] = max(0.0, session_params["omission_prob"] - 0.1)
                elif difficulty_change == "random":
                    session_params["omission_prob"] = random.choice(np.arange(0.0, 0.9, 0.1))

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

            # update gains and sync networks
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
                    d_out = model.d_net(obs_t) # [1, n_actions]
                    d_update = d_out[0, action].item()  # get the D update for the action taken

                # get your PID gains α, β (and kp,ki,kd if you want)
                a_t      = torch.tensor([[int(action)]], dtype=torch.long, device=model.device)
                _, _, _, _, beta = model.gain_adapter.get_gains(obs_t, a_t, None)
                q_curr   = model.policy.q_net(obs_t)[0, action].item()
                q_next   = model.policy.q_net_target(next_t).max(dim=1)[0].item()
                td_err   = reward + (0.0 if done else model.gamma * q_next) - q_curr
                z_update = beta * z_prev + model.gain_adapter.alpha * td_err
                _        = model.policy.q_net(obs_t)[0, int(action)].item()
                _        = model.policy.q_net(next_t)[0, int(action)].item()

            # add to the replay buffer
            model.replay_buffer.add(obs=np.array(obs),
                                    next_obs=np.array(next_obs),
                                    action=np.array([action]),
                                    reward=np.array([reward], dtype=np.float32),
                                    done=done,
                                    infos=[info],
                                    d=np.array([d_update], dtype=np.float32),
                                    z=np.array([z_update], dtype=np.float32),
                                    )
            # record every timestep in the session trace
            recorder.record_env_step(trial_idx, action, reward, next_obs, info, model=model)
            # update obs, z_prev
            obs, z_prev = next_obs, z_update

        # update exploration rate upon trial completion
        if outcome == "trial_end":
            trial_idx += 1  # update trial index
            enl_count = 0   # reset ENL count
            frac = min(1.0, trial_idx / max(1, decay_trials))
            eps = pid_params["initial_eps"] + frac * (pid_params["minimum_eps"] - pid_params["initial_eps"])
            pbar.update(1)
        else:
            # punish if stuck in ENL for > 200 steps
            enl_count = enl_count + 1
            reward -= max(enl_count - session_params["enl_threshold"], 0) * session_params["enl_punish_scale"]
            # reset the seed and retrain if ENL > 1000 steps
            if enl_count > 500:
                retrain = True
                print(f"ENL break after {enl_count} steps, retraining with different seed...")
                return recorder, retrain

        # train
        model.train(batch_size=session_params["batch_size"],
                    seq_len=trial_timesteps,
                    gradient_steps=session_params["gradient_steps"])

        # restore buffer & advance
        model.replay_buffer = orig_buffer

    pbar.close()

    return recorder, retrain


# ----------------------------------------------------------------
# Main execution block for running the sweep
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Define sweep grid
    kd_values        = [0]  # PID derivative gain values
    omission_probs   = [0]
    repeats          = 1  # Number of repeats for each combination


    # Save results settings
    batch_name = 'kd_omission_sweep'
    os.makedirs("PID-results", exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y%m%d")
    os.makedirs(f"PID-results/{today}-{batch_name}", exist_ok=True)
    results = {}

    # Prevent CUDA from being used (patch)
    import TabularPID.Agents.DQN.DQN_policy as _dp
    # find whatever class has jump_start_cuda and override it
    for _name in dir(_dp):
        cls = getattr(_dp, _name)
        if isinstance(cls, type) and hasattr(cls, "jump_start_cuda"):
            cls.jump_start_cuda = lambda self: None

    # Loop through all combinations of kd and omission_prob
    for kd, omit in itertools.product(kd_values, omission_probs):
        # Create copies of the base parameters for each sweep iteration
        sp = copy.deepcopy(base_session_params)
        pp = copy.deepcopy(base_pid_params)
        sp["omission_prob"] = omit
        pp["kd"]             = kd

        print(f"\n=== Running sweep: kd={kd}, omission_prob={omit} ===")
        for r in range(repeats):
            retrain = True
            while retrain:
                # Set global seed for reproducibility
                new_seed = random.randint(0, 10000)
                set_global_seeds(new_seed)
                pp["seed"] = new_seed

                # Train once with the current parameters
                print(f"Training with kd={kd}, omission_prob={omit} (repeat {r + 1}/{repeats})")
                rec, retrain = train_once(sp, pp)

            # Plot and save summary figure
            plot_figure(rec, dt=sp["dt"], pre_steps=sp["pre_steps"], post_steps=sp["post_steps"], show=True,
                        save=True, save_path=f"PID-results/{today}-{batch_name}/kd_{kd}_omit_{omit}_seed_{pp['seed']}.png")

            # Store both params and recorder
            results[(kd, omit, r)] = {
                "session_params": sp,
                "pid_params":     pp,
                "recorder":       rec,
                "seed":           pp["seed"],
            }

        # Save everything
        with open(f"PID-results/{today}-{batch_name}/results_Kd_{kd}_omit_{omit}.pkl", "wb") as f:
            pickle.dump(results, f)

    print(f"\nAll sweeps completed. Results saved to PID-results/{today}-{batch_name}")
