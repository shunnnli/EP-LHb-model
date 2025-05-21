import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch.nn.functional as F
import os
from scipy.ndimage import uniform_filter1d
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# Set seeds for reproducibility

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------- Configuration Section ---------- #
config = {
    'render': True,                 # Toggle rendering
    'env_name': "CartPole-v1",      # Environment name
    # 'env_name': "CliffWalking-v0",   # Environment name (unfinished)
    'rules': ['PID'],           # List of rules to use

    'n_networks': 1,                 # Number of networks to train
    'n_episodes': 500,               # Number of episodes to train
    'sample_rate': 50,               # 50hz
    
    'lr_ctxbg': 1e-3,                # Learning rate for CtxBG
    'lr_q_net': 1e-3,                # Learning rate for QNetwork
    'lr_eplhb': 5e-3,                # Learning rate for EPLHb
    'gamma': 0.99,                   # Discount factor

    'noise_min': 1.0,                # Minimum noise for EPLHb
    'noise_max': 1.0,                # Maximum noise for EPLHb

    'compressed_dim': 16,            # Number of CtxBG neurons
    'eplhb_hidden_dim': 32,          # Number of hidden neurons in EPLHb
    'qnet_dim': 64,                  # Number of neurons in QNetwork

    'epsilon_start': 1.0,            # Initial exploration probability
    'epsilon_min': 0.01,             # Minimum exploration probability
    'decay_rate': 0.01,              # Exponential decay rate for epsilon
    'target_update_freq': 1,         # Episodes between target network updates

    'buffer_size': 10000,            # Replay buffer capacity
    'batch_size': 64,                # Mini-batch size for updates
    'warmup_size': 100,              # Minimum experiences before learning
    'tau': 1e-3,                     # Soft update coefficient (if used)
    'n_step': 1,                     # Number of steps for multi-step returns (1 is still the best)

    'alpha': 0.05,                   # weight on δ′ inside the integral term
    'beta': 0.95,                    # weight on z_pred inside the integral term
    'eta_gain': 1e-7,                # step size η for adapting kp, ki, kd
    'epsilon_gain': 1e-1,            # small ε to avoid div by zero in denom
    'lambda_BR': 0.5,                # λ for running_BR update
}
seed_everything(0)

# ---------- Utility Functions ---------- #
def soft_update(target, source, tau):
    """Soft-update target network parameters."""
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)

def plotSEM(x, y, label=None, color=None, ax=None, alpha=0.2):
    """Plot with shaded error margin."""
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if label is None:
        label = ax._get_lines.get_next_label()
    
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color,
                     edgecolor='None', label='_nolegend_')
    

def fft(td_error, sample_rate, n_networks=None):
    all_magnitudes = []

    for i in range(n_networks):
        signal = td_error[i]
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]

        magnitudes = np.abs(fft_vals)
        magnitudes[1:] *= 2
        all_magnitudes.append(magnitudes)

    all_magnitudes = np.stack(all_magnitudes)

    return freqs, all_magnitudes
    
def plot_results(col, rule, n_networks=1):
    # Grab your histories (shape: [n_nets, episodes])
    data_r = plot_dict[rule]['reward_histories']
    data_l = plot_dict[rule]['loss_histories']
    
    _, episodes = data_r.shape
    fontsize = 8

    # Plot reward and loss histories
    ax0 = axs[col, 0]
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Total Reward", color='tab:blue')
    plotSEM(np.arange(n_episodes), data_r, label='Reward', color='tab:blue', ax=ax0)
    ax0.axhline(195, color='black', linestyle='--')
    ax0.tick_params(axis='y', labelcolor='tab:blue')
    ax1 = ax0.twinx()
    ax1.set_ylabel("Loss", color='tab:red')
    plotSEM(np.arange(n_episodes), data_l, label='Loss', color='tab:red', ax=ax1)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax0.set_title(f"Rule: {rule} | Avg reward: {np.mean(avg_reward):.2f}", fontsize=fontsize)

    # # Plot TD-errors
    ax2 = axs[col, 1]
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Value")
    plotSEM(np.arange(episodes), plot_dict[rule]['td_error_histories'], label='TD-error', color='tab:green', ax=ax2)
    plotSEM(np.arange(episodes), plot_dict[rule]['final_error_histories'], label='Final-error', color='tab:orange', ax=ax2)
    ax3 = ax2.twinx()
    ax3.set_ylabel("Delta TD error", color='tab:purple')
    ax3.tick_params(axis='y', labelcolor='tab:purple')
    plotSEM(np.arange(episodes), plot_dict[rule]['delta_td_histories'], label='Delta TD error', color='tab:purple', ax=ax3)
    ax2.legend()
    ax2.set_title("TD Errors Over Time", fontsize=fontsize)

    ax4 = axs[col, 2]
    ax4.set_xlabel("Frequency")
    ax4.set_ylabel("Magnitude")
    for rule in config['rules']:
        if rule == 'PID':
            color = 'tab:blue'
        elif rule == 'TD':
            color = 'tab:orange'
        else:
            color = 'tab:green'
        freqs, magnitudes = fft(td_error = plot_dict[rule]['final_error_histories'], sample_rate = config['sample_rate'], n_networks = n_networks)
        plotSEM(freqs, magnitudes, label=rule, color=color, ax=ax4)

    # Compute significance
    if n_networks >= 2 and len(config['rules']) > 1:
        td_freqs, td_magnitudes = fft(td_error = plot_dict['TD']['final_error_histories'], sample_rate = config['sample_rate'], n_networks = n_networks)
        pd_freqs, pd_magnitudes = fft(td_error = plot_dict['PID']['final_error_histories'], sample_rate = config['sample_rate'], n_networks = n_networks) 
        t_vals, p_vals = ttest_rel(td_magnitudes, pd_magnitudes, axis=0)
        _, pvals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
        signif_mask = pvals_corrected < 0.05

        td_magnitudes_mean = np.mean(td_magnitudes, axis=0)
        ax4.scatter(td_freqs[signif_mask],
                    td_magnitudes_mean[signif_mask],
                    color='red', s=10, label='p < 0.05')
        ax4.set_title("FFT: Final TD Error Signal", fontsize=fontsize)
        ax4.set_xlim(0, (config['sample_rate'])/2)
        ax4.legend()
    else:
        print("Not enough networks to compute significance.")
        ax4.legend()



class ReplayBuffer:
    """Simple FIFO replay buffer"""
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.stack(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.stack(next_state),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ---------- Biological Neural Network Modules ---------- #
class CtxBG(nn.Module):
    """Models cortical + basal ganglia compression."""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        # Existing cortex-basal ganglia encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config['compressed_dim']),
            nn.ReLU()
        )
        # Forward model head: predict next comperssed state from z and action
        self.forward_model = nn.Sequential(
            nn.Linear(config['compressed_dim'] + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config['compressed_dim']),
            nn.ReLU()
        )
    def encode(self, obs):
        # returns compressed state z
        return self.encoder(obs)

    def predict(self, z, a):
        # a: LongTensor of shape [batch]
        a_onehot = F.one_hot(a, num_classes=self.forward_model[0].in_features - z.size(1)).float()
        inp = torch.cat([z, a_onehot], dim=-1)
        return self.forward_model(inp)

class QNetwork(nn.Module):
    """Q(s,a) network that takes compressed features as input."""
    def __init__(self, compressed_dim, action_dim):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(compressed_dim, config['qnet_dim']),
            nn.ReLU(),
            nn.Linear(config['qnet_dim'], action_dim)
        )
    def forward(self, x):
        return self.q_net(x)

class EPLHb(nn.Module):
    """EP–LHb see both the current and predicted value–features and learn to compare them itself."""
    def __init__(self, compressed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*compressed_dim, config['eplhb_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['eplhb_hidden_dim'], 1)
        )
    def forward(self, z, z_pred=None):
        # x = torch.cat([z, td_error.unsqueeze(-1)], dim=-1)
        if z_pred is None:
            z_pred = torch.zeros_like(z)
        x = torch.cat([z, z_pred], dim=-1)
        return self.net(x).squeeze(-1)

# ---------- RL Agent with Multi-step Returns & Target Network ---------- #
class BioQAgent:
    def __init__(self, obs_dim, action_dim, device="cpu"):
        self.device = device
        # Initialize CtxBG
        self.ctxbg = CtxBG(obs_dim,action_dim).to(device)
        # Initialize Q-network and target network
        self.q_net = QNetwork(config['compressed_dim'], action_dim).to(device)
        self.q_target = QNetwork(config['compressed_dim'], action_dim).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()
        # Initialize EPLHb
        self.eplhb = EPLHb(config['compressed_dim']).to(device)
        # Initialize optimizer
        self.optimizer = optim.Adam([
            {'params': self.ctxbg.parameters(), 'lr': config['lr_ctxbg']},
            {'params': self.q_net.parameters(),  'lr': config['lr_q_net']},
            {'params': self.eplhb.parameters(), 'lr': config['lr_eplhb']},
        ])

        # Set PID gains
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.0
        # Running estimate of <δ²> for normalization
        self.running_BR = 0.0
        # store previous Q-value (for derivative term)
        self.prev_q_val = None

        self.prev_td_error = torch.zeros(config['batch_size'], device=device) # Track previous td-error
        self.buffer = ReplayBuffer(config['buffer_size'], config['batch_size'])
        self.n_step_buffer = deque(maxlen=config['n_step'])
        self.action_dim = action_dim
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.n_step = config['n_step']
        self.tau = config['tau']

    def update_target_network(self):
        if hasattr(self, 'tau') and self.tau > 0:
            soft_update(self.q_target, self.q_net, config['tau'])
        else:
            self.q_target.load_state_dict(self.q_net.state_dict())

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z = self.ctxbg.encode(obs_t)
            q_values = self.q_net(z)
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        return torch.argmax(q_values).item()

    def _get_n_step_info(self):
        R = 0.0
        next_state, done = None, False
        for idx, (_s, _a, r, s_next, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            next_state = s_next
            done = d
            if d:
                break
        return R, next_state, done

    def store(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            R, s_n, d_n = self._get_n_step_info()
            s0, a0, _, _, _ = self.n_step_buffer[0]
            self.buffer.push(s0, a0, R, s_n, d_n)

    def finish_episode(self):
        while len(self.n_step_buffer) > 0:
            R, s_n, d_n = self._get_n_step_info()
            s0, a0, _, _, _ = self.n_step_buffer[0]
            self.buffer.push(s0, a0, R, s_n, d_n)
            self.n_step_buffer.popleft()

    def learn(self):
        if len(self.buffer) < config['warmup_size']:
            return {
                'loss':            0.0,
                'td_error':        0.0,
                'td_error_noised': 0.0,
                'delta_td':        0.0,
                'final_td_error':  0.0,
                'eplhb_out':       0.0,
            }
        states, actions, rewards, next_states, dones = self.buffer.sample()
        state_t      = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_state_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        action_t     = torch.tensor(actions, dtype=torch.int64).to(self.device)
        reward_t     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done_t       = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Encode current and next compressed states
        z      = self.ctxbg.encode(state_t)
        next_z = self.ctxbg.encode(next_state_t)
        # Compute Q-values for current state action pair
        q_vals = self.q_net(z)
        current_q  = q_vals.gather(1, action_t.unsqueeze(1)).squeeze(1)
        
        # Feedback TD error, Line 7
        with torch.no_grad():
            next_q    = self.q_target(next_z)
            max_next  = next_q.max(1)[0]
            target_q = reward_t + (self.gamma ** self.n_step) * max_next * (1 - done_t)
        target_q = target_q.detach()
        td_error  = target_q - current_q

        # Feedforward TD error, Line 6
        with torch.no_grad():
            z_pred = self.ctxbg.predict(z, action_t)
            next_q_pred = self.q_target(z_pred)
            max_next_pred = next_q_pred.max(1)[0]
            target_q_pred = reward_t + (self.gamma ** self.n_step) * max_next_pred * (1 - done_t)
        td_error_pred = target_q_pred - current_q.detach()

        # Feedforward D-term
        delta_td = td_error_pred - self.prev_td_error
        self.prev_td_error = td_error_pred.detach()

        # EP LHb to calculate D-term
        eplhb_out = self.eplhb(z, z_pred)
        
        if config['rule'] == 'EPLHb':
            # Use EPLHb to modulate the TD-error
            EPLHb_coeff = -torch.sigmoid(self.EPLHb_coeff)
            noise = torch.empty_like(td_error).uniform_(config['noise_min'], config['noise_max'])
            final_td_error = EPLHb_coeff * eplhb_out + noise * td_error
        elif config['rule'] == 'PID':
            # Line 8: gain updates
            denom = self.running_BR + config['epsilon_gain']
            # κp ← κp + η·(δ·δ') / denom
            self.kp += config['eta_gain'] * (td_error * td_error_pred).mean().item() / denom
            # κI ← κI + η·(δ·I_input) / denom
            I_input = config['beta'] * z_pred.mean(dim=1) + config['alpha'] * td_error_pred
            self.ki += config['eta_gain'] * (td_error * I_input).mean().item() / denom
            # κd ← κd + η·(δ·ΔQ) / denom, where ΔQ = Q(s,a) − prev_Q
            delta_q = (current_q.detach() - self.prev_q_val if self.prev_q_val is not None else torch.zeros_like(current_q))
            self.kd += config['eta_gain'] * (td_error * delta_q).mean().item() / denom

            # Line 9: Q value update
            # proportional term
            P_term = self.kp * td_error
            # integral term input = β·z_pred + α·δ'
            I_term  = self.ki * I_input
            # derivative term
            D_term = self.kd * delta_td
            # combine P+I+D
            combined = P_term + I_term + D_term
            # Add optional noise
            noise = torch.empty_like(combined).uniform_(config['noise_min'], config['noise_max'])
            final_td_error = noise * combined

            # Line 11: update params
            # update running variance: running_BR ← (1−λ)·running_BR + λ·δ²
            self.running_BR = (
                (1.0 - config['lambda_BR']) * self.running_BR
                + config['lambda_BR'] * td_error.detach().pow(2).mean().item()
            )
            self.prev_q_val = current_q.detach().clone() # save for next step

        else:
            # Use standard TD-error
            noise = torch.empty_like(td_error).uniform_(config['noise_min'], config['noise_max'])
            final_td_error = noise * td_error

        loss = final_td_error.pow(2).mean()
        # loss = F.smooth_l1_loss(final_td_error, torch.zeros_like(final_td_error))
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100) # In-place gradient clipping
        self.optimizer.step()
        self.update_target_network()

        return {
            'loss':       loss.item(),
            'td_error':   td_error.detach().mean().item(),
            'td_error_noised': (td_error * noise).detach().mean().item(),
            'delta_td':  delta_td.detach().mean().item(),
            'final_td_error':  final_td_error.detach().mean().item(),
            'eplhb_out':  eplhb_out.detach().mean().item(),
        }

# ---------- Training Loop ---------- #
def one_hot(s, n):
    v = np.zeros(n, dtype=np.float32)
    v[s] = 1.0
    return v

def train(env_name="CartPole-v1", episodes=500):
    env = gym.make(env_name)

    is_discrete = isinstance(env.observation_space, gym.spaces.Discrete)
    n_states = env.observation_space.n if is_discrete else None
    obs_dim  = n_states if is_discrete else env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the agent
    agent = BioQAgent(obs_dim, action_dim)
    
    best_reward = -float('inf')
    reward_history = []
    loss_history = []
    td_error_history = []
    td_error_noised_history = []
    delta_td_history = []
    final_td_error_history = []
    eplhb_history = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        if is_discrete:
            obs = one_hot(obs, n_states)
        done = False
        total_reward = 0
        agent.n_step_buffer.clear()

        # Exponential annealing for epsilon
        agent.epsilon = max(
            agent.epsilon_min,
            agent.epsilon_min + (config['epsilon_start'] - agent.epsilon_min)
            * np.exp(-config['decay_rate'] * (ep - 1))
        )

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            results = agent.learn()

            # if ep % config['target_update_freq'] == 0:
            #     agent.update_target_network()

            if is_discrete: 
                next_obs = one_hot(next_obs, n_states)
            obs = next_obs
            total_reward += reward
        agent.finish_episode()

        if results is not None:
            reward_history.append(total_reward)
            loss_history.append(results['loss'])
            td_error_history.append(results['td_error'])
            td_error_noised_history.append(results['td_error_noised'])
            delta_td_history.append(results['delta_td'])
            final_td_error_history.append(results['final_td_error'])
            eplhb_history.append(results['eplhb_out'])

        if total_reward > best_reward:
            best_reward = total_reward

            if config['render']:
                # Save the model if needed
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                save_dir = "checkpoints"
                torch.save({
                    "ctxbg":    agent.ctxbg.state_dict(),
                    "q_net":    agent.q_net.state_dict(),
                    "q_target": agent.q_target.state_dict(),
                    "eplhb":    agent.eplhb.state_dict(),
                }, os.path.join(save_dir, "best_model.pth"))

        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes} | Total Reward: {total_reward} | Loss: {results['loss']:.4f} | Final TD Error: {results['final_td_error']:.4f} | Rule: {config['rule']}")

    env.close()

    # 3) At the end, return the trained agent (latest weights)
    results_dict = {
        'rewards':          reward_history,
        'losses':           loss_history,
        'td_errors':        td_error_history,
        'td_errors_noised': td_error_noised_history,
        'delta_td_errors':  delta_td_history,
        'final_td_errors':  final_td_error_history,
        'eplhb_outputs':    eplhb_history,
    }
    return agent, results_dict
    

# ---------- Main Execution ---------- #
if __name__ == "__main__":

    plot_dict = {}
    best_reward = -float('inf')
    best_model = None
    best_model_rule = None
    best_model_path = "checkpoints/best_model.pth"

    n_networks = config['n_networks']
    n_episodes = config['n_episodes']
    env_name = config['env_name']

    for rule in config['rules']:
        config['rule'] = rule

        # Initialize history records
        avg_reward = np.zeros(n_networks)
        reward_histories = np.zeros((n_networks, n_episodes))
        loss_histories = np.zeros((n_networks, n_episodes))
        td_error_histories = np.zeros((n_networks, n_episodes))
        td_error_noised_histories = np.zeros((n_networks, n_episodes))
        delta_td_histories = np.zeros((n_networks, n_episodes))
        final_error_histories = np.zeros((n_networks, n_episodes))
        eplhb_histories = np.zeros((n_networks, n_episodes))

        for i in range(n_networks):
            print(f"Training network {i+1}/{n_networks} | Rule: {rule}")
            # Train the agent
            agent, train_results = train(env_name=env_name, episodes=n_episodes)
            avg_reward[i] = np.mean(train_results['rewards'][-100:])
            reward_histories[i] = np.array(train_results['rewards'])
            loss_histories[i] = np.array(train_results['losses'])
            td_error_histories[i] = np.array(train_results['td_errors'])
            td_error_noised_histories[i] = np.array(train_results['td_errors_noised'])
            delta_td_histories[i] = np.array(train_results['delta_td_errors'])
            final_error_histories[i] = np.array(train_results['final_td_errors'])
            eplhb_histories[i] = np.array(train_results['eplhb_outputs'])

            # Save the model if needed if its the best model
            if avg_reward[i] > best_reward:
                best_reward = avg_reward[i]
                best_model = agent
                best_model_rule = rule
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                torch.save({
                    "ctxbg":    agent.ctxbg.state_dict(),
                    "q_net":    agent.q_net.state_dict(),
                    "q_target": agent.q_target.state_dict(),
                    "eplhb":    agent.eplhb.state_dict(),
                    "rule":     rule,
                    "avg_reward": avg_reward[i],
                }, best_model_path)
                print(f"Saved best model to {best_model_path}")
            print(f"Average reward for network {i+1}: {avg_reward[i]:.2f}")

        # save each rule's results
        plot_dict[rule] = {
                'avg_reward':               avg_reward,
                'reward_histories':         reward_histories,
                'loss_histories':           loss_histories,
                'td_error_histories':       td_error_histories,
                'final_error_histories':    final_error_histories,
                'delta_td_histories':       delta_td_histories,
                'eplhb_histories':          eplhb_histories,
                'td_error_noised_histories':td_error_noised_histories}

        # ─── 0) Print the average reward for each networks ────────────────────────────────
        print("Average rewards for each network:")
        for i in range(n_networks):
            print(f"Network {i+1}: {avg_reward[i]:.2f} | Rule: {rule}")

        # Save the results
        np.savez("results_" + rule + ".npz",
                rewards=reward_histories,
                losses=loss_histories,
                td_errors=td_error_histories,
                td_errors_noised=td_error_noised_histories,
                delta_td_errors=delta_td_histories,
                final_td_errors=final_error_histories,
                eplhb_outputs=eplhb_histories)

    # ─── 1) Plot the training results ────────────────────────────────
    fig, axs = plt.subplots(len(config['rules']), 3, figsize=(20, 10))
    if len(config['rules']) == 1:
        axs = axs[np.newaxis, :]
    for i in range(len(config['rules'])):
        plot_results(i, config['rules'][i],n_networks=n_networks)
    fig.tight_layout()
    plt.show()

    # ─── 3) Render a few episodes with the “best” agent ────────────────────────
    if config['render']:
        render_env = gym.make("CartPole-v1", render_mode="human")
        
        # Reload the best model
        ckpt = torch.load("checkpoints/best_model.pth")
        agent.ctxbg.load_state_dict(ckpt["ctxbg"])
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.q_target.load_state_dict(ckpt["q_target"])
        agent.eplhb.load_state_dict(ckpt["eplhb"])
        best_model_rule = ckpt["rule"]
        best_model_avg_reward = ckpt["avg_reward"]
        print(f"Loaded best model with avg reward {best_model_avg_reward:.2f} | Rule: {best_model_rule}")
        # set to eval mode
        agent.ctxbg.eval()
        agent.q_net.eval()
        agent.q_target.eval()
        agent.eplhb.eval()

        for ep in range(5):
            obs, _ = render_env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(obs)
                obs, reward, done, _, _ = render_env.step(action)
                total_reward += reward
                # for some gym versions you may also need:
                # render_env.render()
            print(f"[Render] Episode {ep+1} → Reward {total_reward:.1f}")
        render_env.close()