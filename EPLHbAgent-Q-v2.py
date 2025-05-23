import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
# import torch.nn.functional as F
import os

# Set seeds for reproducibility

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------- Configuration Section ---------- #
config = {
    'render': False,                 # Toggle live rendering
    'with_EPLHb': True,              # Use EPLHb module
    
    'lr_ctxbg': 1e-4,                # Learning rate for CtxBG
    'lr_q_net': 1e-4,                # Learning rate for QNetwork
    'lr_eplhb': 1e-4,                # Learning rate for EPLHb
    'gamma': 0.99,                   # Discount factor

    'noise_min': 1,                # Minimum noise for EPLHb
    'noise_max': 1,                # Maximum noise for EPLHb

    'compressed_dim': 16,            # Number of CtxBG neurons
    'eplhb_hidden_dim': 32,          # Number of hidden neurons in EPLHb
    'qnet_dim': 64,                  # Number of neurons in QNetwork

    'epsilon_start': 1.0,            # Initial exploration probability
    'epsilon_min': 0.10,             # Minimum exploration probability
    'decay_rate': 0.01,              # Exponential decay rate for epsilon
    'target_update_freq': 1,        # Episodes between target network updates

    'buffer_size': 10000,            # Replay buffer capacity
    'batch_size': 64,                # Mini-batch size for updates
    'warmup_size': 100,              # Minimum experiences before learning
    'tau': 5e-3,                     # Soft update coefficient (if used)
    'n_step': 1                      # Number of steps for multi-step returns (1 is still the best)
}

seed_everything(0)

# ---------- Utility Functions ---------- #
def soft_update(target, source, tau):
    """Soft-update target network parameters."""
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)

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
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config['compressed_dim']),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

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
    """EP–LHb synapse that shapes the TD-error."""
    def __init__(self, compressed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(compressed_dim, config['eplhb_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['eplhb_hidden_dim'], 1)
        )
    def forward(self, z):
        # x = torch.cat([z, td_error.unsqueeze(-1)], dim=-1)
        x = z
        return self.net(x).squeeze()

# ---------- RL Agent with Multi-step Returns & Target Network ---------- #
class BioQAgent:
    def __init__(self, obs_dim, action_dim, device="cpu"):
        self.device = device
        self.ctxbg = CtxBG(obs_dim).to(device)
        self.q_net = QNetwork(config['compressed_dim'], action_dim).to(device)
        self.q_target = QNetwork(config['compressed_dim'], action_dim).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()
        self.eplhb = EPLHb(config['compressed_dim']).to(device)
        

        self.EPLHb_coeff = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32, device=device))

        self.optimizer = optim.Adam([
            {'params': self.ctxbg.parameters(), 'lr': config['lr_ctxbg']},
            {'params': self.q_net.parameters(),  'lr': config['lr_q_net']},
            {'params': self.eplhb.parameters(), 'lr': config['lr_eplhb']},
            {'params': self.EPLHb_coeff, 'lr': config['lr_eplhb']}
        ])
        self.buffer = ReplayBuffer(config['buffer_size'], config['batch_size'])
        self.n_step_buffer = deque(maxlen=config['n_step'])
        self.action_dim = action_dim
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.n_step = config['n_step']

    def update_target_network(self):
        if hasattr(self, 'tau') and self.tau > 0:
            soft_update(self.q_target, self.q_net, config['tau'])
        else:
            self.q_target.load_state_dict(self.q_net.state_dict())

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z = self.ctxbg(obs_t)
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
                'final_td_error':  0.0,
                'eplhb_out':       0.0,
            }
        states, actions, rewards, next_states, dones = self.buffer.sample()
        state_t      = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_state_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        action_t     = torch.tensor(actions, dtype=torch.int64).to(self.device)
        reward_t     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done_t       = torch.tensor(dones, dtype=torch.float32).to(self.device)
        z      = self.ctxbg(state_t)
        next_z = self.ctxbg(next_state_t)
        q_vals = self.q_net(z)
        current_q_val  = q_vals.gather(1, action_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q    = self.q_target(next_z)
            max_next  = next_q.max(1)[0]
            target_q_val = reward_t + (self.gamma ** self.n_step) * max_next * (1 - done_t)
        target_q_val = target_q_val.detach()
        td_error  = target_q_val - current_q_val
        eplhb_out = self.eplhb(z)
        
        if config['with_EPLHb']:
            # Use EPLHb to modulate the TD-error
            EPLHb_coeff = -torch.sigmoid(self.EPLHb_coeff)
            noise = torch.empty_like(td_error).uniform_(config['noise_min'], config['noise_max'])
            final_td_error = EPLHb_coeff * eplhb_out + noise * td_error
        else:
            # Use standard TD-error
            noise = torch.empty_like(td_error).uniform_(1, 1)
            final_td_error = td_error

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
            'final_td_error':  final_td_error.detach().mean().item(),
            'eplhb_out':  eplhb_out.detach().mean().item(),
        }

# ---------- Training Loop ---------- #
def train(env_name="CartPole-v1", episodes=500):
    env = gym.make(env_name, render_mode="human" if config['render'] else None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = BioQAgent(obs_dim, action_dim)
    
    best_reward = -float('inf')
    reward_history = []
    loss_history = []
    td_error_history = []
    td_error_noised_history = []
    final_td_error_history = []
    eplhb_history = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
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
            if config['render']:
                env.render()
            action = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            results = agent.learn()

            # if ep % config['target_update_freq'] == 0:
            #     agent.update_target_network()

            obs = next_obs
            total_reward += reward
        agent.finish_episode()

        if results is not None:
            reward_history.append(total_reward)
            loss_history.append(results['loss'])
            td_error_history.append(results['td_error'])
            td_error_noised_history.append(results['td_error_noised'])
            final_td_error_history.append(results['final_td_error'])
            eplhb_history.append(results['eplhb_out'])

        if total_reward > best_reward:
            best_reward = total_reward
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
            print(f"Episode {ep}/{episodes} | Total Reward: {total_reward} | Loss: {results['loss']:.4f} | Final TD Error: {results['final_td_error']:.4f}")

    env.close()

    # 3) At the end, return the trained agent (latest weights)
    results_dict = {
        'rewards':          reward_history,
        'losses':           loss_history,
        'td_errors':        td_error_history,
        'td_errors_noised': td_error_noised_history,
        'final_td_errors':  final_td_error_history,
        'eplhb_outputs':    eplhb_history,
    }
    return agent, results_dict
    

# ---------- Main Execution ---------- #
if __name__ == "__main__":
    agent, train_results = train()

    # ─── 1) Plot the training results ────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex=True)

    # Plot reward and loss history on two y-axes
    ax0 = axs[0]
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Total Reward", color='tab:blue')
    ax0.plot(train_results['rewards'], color='tab:blue')
    ax0.tick_params(axis='y', labelcolor='tab:blue')
    ax1 = ax0.twinx()
    ax1.set_ylabel("Loss", color='tab:red')
    ax1.plot(train_results['losses'], color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    avg_reward = np.mean(train_results['rewards'])
    avg_loss   = np.mean(train_results['losses'])
    ax0.set_title(f"Avg reward: {avg_reward:.2f} | Avg loss: {avg_loss:.2f}")

    # Plot TD-error and EPLHb output
    ax2 = axs[1]
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Value")
    ax2.plot(train_results['td_errors'], label='TD-error')
    ax2.plot(train_results['td_errors_noised'], label='Noised TD-error')
    ax2.plot(train_results['final_td_errors'], label='Final TD-error')
    ax2.plot(train_results['eplhb_outputs'], label='EPLHb Output')
    ax2.legend(["TD error (raw)", "TD error (noised)","TD error (final)", "EPLHb Output"])
    ax2.set_title("Errors and EPLHb Output Over Time")

    # Plot differences between TD error versions
    ax3 = axs[2]
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Value")
    ax3.plot(np.array(train_results['td_errors']) - np.array(train_results['td_errors_noised']), label='TD-error - Noised TD-error')
    ax3.plot(np.array(train_results['td_errors']) - np.array(train_results['final_td_errors']), label='TD-error - Final TD-error')
    ax3.legend(["TD error - Noised TD-error", "TD error - Final TD-error"])
    ax3.set_title("Differences Between TD Error Versions")
    fig.tight_layout()
    plt.show()

    # ─── 2) Reload the best model ────────────────────────────────
    ckpt = torch.load("checkpoints/best_model.pth")
    agent.ctxbg.load_state_dict(ckpt["ctxbg"])
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.q_target.load_state_dict(ckpt["q_target"])
    agent.eplhb.load_state_dict(ckpt["eplhb"])

    # set to eval mode
    agent.ctxbg.eval()
    agent.q_net.eval()
    agent.q_target.eval()
    agent.eplhb.eval()

    # ─── 3) Render a few episodes with the “best” agent ────────────────────────
    render_env = gym.make("CartPole-v1", render_mode="human")
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