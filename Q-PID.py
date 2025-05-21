#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
import sys
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
from TabularPID.MDPs.Policy    import Policy
from TabularPID.Agents.Agents  import ControlledQLearning, learning_rate_function

def plotSEM(x, y, label=None, color=None, ax=None, alpha=0.2):
    """Plot with shaded error margin."""
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if label is None:
        label = ax._get_lines.get_next_label()

    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color,
                     edgecolor='None', label='_nolegend_')

# ─── 3) Wrap Gym’s CliffWalking-v0 to match their Agent API ──────────────
class GymCliffEnv:
    def __init__(self, seed=0):
        self.env = gym.make("CliffWalking-v0", render_mode=None)
        self.num_states  = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        # their Policy class needs a PRNG
        self.prg = np.random.RandomState(seed)

    def reset(self):
        obs, _ = self.env.reset()
        self.current_state = obs
        return obs

    def take_action(self, action):
        # step the Gym env, then clamp the −100 “cliff” penalty to −1
        next_s, reward, done, truncated, info = self.env.step(action)
        reward = max(reward, -1.0)
        self.current_state = next_s
        return next_s, reward

    def set_seed(self, seed):
        self.env.reset(seed=seed)
        self.prg.seed(seed)

# ─── 4) Instantiate environment, policy, and two agents ──────────────────
env = GymCliffEnv(seed=42)
policy = Policy(env.num_actions, env.num_states, env.prg, None)

# Common hyper‐params from Figure 7 (control experiments)
gamma = 0.99
kp, ki, kd = 1.0, 0.0, 0.0       # for pure Q‐Learning baseline
pid_kp, pid_ki, pid_kd = 1.0, 1.0, 1.0  # you can tune these for PID

alpha = 0.05
beta  = 0.95

# learning‐rate schedules for P, I, and D per the author’s code:
lr_P = learning_rate_function(alpha, np.inf)  # constant α
lr_I = learning_rate_function(alpha, np.inf)
lr_D = learning_rate_function(alpha, np.inf)

# 4a) pure‐Q‐Learning baseline
agent_q = ControlledQLearning(
    environment=env,
    gamma=gamma,
    kp=kp, ki=ki, kd=kd,
    alpha=alpha,
    beta=beta,
    learning_rate=(lr_P, lr_I, lr_D),
    double=False
)

# 4b) PID‐Accelerated TD‐Learning
agent_pid = ControlledQLearning(
    environment=env,
    gamma=gamma,
    kp=pid_kp, ki=pid_ki, kd=pid_kd,
    alpha=alpha,
    beta=beta,
    learning_rate=(lr_P, lr_I, lr_D),
    double=False
)

# ─── 5) Run both agents for 20 000 steps & record value‐error ────────────
max_steps = 200000
n_runs = 1

def make_value_error_test(env):
    # We’ll compare max_a Q(s,a) to the true V*(s) under “always‐right” policy.
    # Compute V_true by synchronous policy evaluation:
    raw = env.env.unwrapped
    # build deterministic “always‐right” policy
    optimal_pi = {
        s: max(raw.P[s].keys(),
               key=lambda a: sum(p*(r+gamma*0) for (p,_,r,_) in raw.P[s][a]))
        for s in range(env.num_states)
    }
    # policy‐evaluation
    V = np.zeros(env.num_states)
    while True:
        δ_max = 0
        for s in range(env.num_states):
            a = optimal_pi[s]
            v_new = sum(
                p * (r + gamma * V[s2] * (1-d))
                for (p, s2, r, d) in raw.P[s][a]
            )
            δ_max = max(δ_max, abs(v_new - V[s]))
            V[s] = v_new
        if δ_max < 1e-6:
            break
    V_norm = np.linalg.norm(V, ord=1)
    def test_fn(Q, Qp, BR):
        Vt = Q.max(axis=1)
        return np.linalg.norm(Vt - V, ord=1) / V_norm
    return test_fn

test_fn = make_value_error_test(env)

hist_q   = np.zeros((n_runs, max_steps))
hist_pid = np.zeros((n_runs, max_steps))

print("Running pure Q-Learning baseline…")
for run in range(n_runs):
    print(f"Run {run+1}/{n_runs} (Q-learning)…")
    # Reset the environment and agent for each run
    env.reset()
    agent_q.reset()

    # Run the agents for max_steps
    h_q, _ = agent_q.estimate_value_function(
        num_iterations=max_steps,
        test_function=test_fn,
        stop_if_diverging=False,
        follow_trajectory=False,
        reset_environment=True
    )
    hist_q[run, :] = h_q

print("Running PID-Accelerated TD-Learning…")
for run in range(n_runs):
    print(f"Run {run+1}/{n_runs} (PID) …")
    # Reset the environment and agent for each run
    env.reset()
    agent_pid.reset()

    # Run the agents for max_steps
    h_pid, _ = agent_pid.estimate_value_function(
        num_iterations=max_steps,
        test_function=test_fn,
        stop_if_diverging=False,
        follow_trajectory=False,
        reset_environment=True
    )
    hist_pid[run, :] = h_pid

# ─── 6) Plot the normalized ‖Vₜ−V*‖₁ curves ─────────────────────────────
ts = np.arange(max_steps)
plt.figure(figsize=(8,4))
plotSEM(ts, hist_q, label="Q-Learning")
plotSEM(ts, hist_pid, label="PID-Accelerated TD-Learning")
plt.xlabel("Steps (t)")
plt.ylabel(r"Normalized $\|V_t - V^*\|_1$")
plt.title("CliffWalking-v0: Q-Learning vs. PID-Accelerated TD")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()