import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import deque

# Set seeds for reproducibility

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------- Configuration Section ---------- #
config = {
    'render': False,                 # Toggle live rendering
    'compressed_dim': 16,            # Number of CtxBG neurons
    'eplhb_hidden_dim': 32,          # Number of hidden neurons in EPLHb
    'qnet_dim': 128,                  # Number of neurons in QNetwork
    'lr_ctxbg': 1e-4,                # Learning rate for CtxBG
    'lr_q_net': 1e-4,                # Learning rate for QNetwork
    'lr_eplhb': 5e-3,                # Learning rate for EPLHb
    'gamma': 0.99,                   # Discount factor
    'epsilon_start': 1.0,            # Initial exploration probability
    'epsilon_min': 0.05,             # Minimum exploration probability
    'decay_rate': 0.01,              # Exponential decay rate for epsilon
    'target_update_freq': 50,        # Episodes between target network updates
    'buffer_size': 10000,            # Replay buffer capacity
    'batch_size': 64,                # Mini-batch size for updates
    'warmup_size': 100,              # Minimum experiences before learning
    'tau': 0.000,                    # Soft update coefficient (if used)
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
    """Advantage‐learning head: estimates V(s) and A(s,a)."""
    def __init__(self, compressed_dim, action_dim):
        super().__init__()
        # shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(compressed_dim, config['qnet_dim']),
            nn.ReLU()
        )
        # state‐value stream
        self.value_stream = nn.Linear(config['qnet_dim'], 1)
        # advantage stream
        self.adv_stream   = nn.Linear(config['qnet_dim'], action_dim)

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f).squeeze(-1)  # V(s), shape [B]
        a = self.adv_stream(f)                # A(s,a), shape [B, A]
        return v, a

class EPLHb(nn.Module):
    """EP–LHb synapse that shapes the TD-error."""
    def __init__(self, compressed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(compressed_dim + 1, config['eplhb_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['eplhb_hidden_dim'], 1)
        )
    def forward(self, z, td_error):
        x = torch.cat([z, td_error.unsqueeze(-1)], dim=-1)
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
        self.optimizer = optim.Adam([
            {'params': self.ctxbg.parameters(), 'lr': config['lr_ctxbg']},
            {'params': self.q_net.parameters(),  'lr': config['lr_q_net']},
            {'params': self.eplhb.parameters(), 'lr': config['lr_eplhb']},
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
        # get shared embedding
        z = self.ctxbg(torch.tensor(obs, dtype=torch.float32).to(self.device))
        # now forward returns (v, a)
        _, a = self.q_net(z)               
        # select greedy on advantage
        if random.random() < self.epsilon:
            return random.randrange(a.size(-1))
        return torch.argmax(a).item()   

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
            return
        # sample a batch
        states, actions, rewards, next_states, dones = self.buffer.sample()
        s  = torch.tensor(states,      dtype=torch.float32).to(self.device)
        ns = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        a_inds = torch.tensor(actions, dtype=torch.int64).to(self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        d  = torch.tensor(dones,       dtype=torch.float32).to(self.device)

        # embeddings
        z     = self.ctxbg(s)
        nz    = self.ctxbg(ns)

        # forward through online & target nets
        v,  a = self.q_net(z)           # v: [B], a: [B,A]
        vt, _ = self.q_target(nz)       # only need v_target

        # compute 1-step TD-target on values
        with torch.no_grad():
            td_target = r + self.gamma * vt * (1 - d)

        # TD-error on V(s)
        td_error = td_target - v # [B]

        # modulate that error via EPLHb
        mod_error = self.eplhb(z, td_error)
        # use the predicted advantage for the action we actually took:
        a_taken = a.gather(1, a_inds.unsqueeze(-1)).squeeze(-1)  # [B]
        
        # final loss: value + advantage via modulated error
        value_loss = F.mse_loss(v, td_target)
        adv_loss   = F.mse_loss(a_taken, mod_error.detach())
        loss = value_loss + adv_loss

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ---------- Training Loop ---------- #
def train(env_name="CartPole-v1", episodes=500):
    env = gym.make(env_name, render_mode="human" if config['render'] else None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = BioQAgent(obs_dim, action_dim)
    reward_history = []
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
            agent.learn()
            if ep % config['target_update_freq'] == 0:
                agent.update_target_network()
            obs = next_obs
            total_reward += reward
        agent.finish_episode()
        reward_history.append(total_reward)
        if ep % 10 == 0:
            print(f"Episode {ep} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}")
    env.close()
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance Over Time")
    plt.show()

if __name__ == "__main__":
    train()
