#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
import sys
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from TabularPID.AgentBuilders.EnvBuilder import get_env_policy
from TabularPID.MDPs.Policy    import Policy
from TabularPID.Agents.Agents  import ControlledQLearning, PID_TD, learning_rate_function

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


# ─── 4) Instantiate environment, policy, and two agents ──────────────────
env_name = "cliff walk"
seed = 0
env, policy = get_env_policy(env_name, seed)

# Common hyper‐params from Figure 7 (control experiments)
gamma = 0.99
kp, ki, kd = 1.0, 0.0, 0.0       # for pure Q‐Learning baseline
pid_kp, pid_ki, pid_kd = 1.0, 0.3, 0.2  # you can tune these for PID

alpha = 0.05
beta  = 0.95

# learning‐rate schedules for P, I, and D per the author’s code:
lr_P = learning_rate_function(alpha, float("inf"))  # constant α
lr_I = learning_rate_function(alpha, float("inf"))
lr_D = learning_rate_function(alpha, float("inf"))

# 4a) pure‐Q‐Learning baseline
agent_q = PID_TD(
    environment=env,
    policy=policy,
    gamma=gamma,
    kp=kp, ki=ki, kd=kd,
    alpha=alpha,
    beta=beta,
    learning_rate=(lr_P, lr_I, lr_D),
)

# 4b) PID‐Accelerated TD‐Learning
agent_pid = PID_TD(
    environment=env,
    policy=policy,
    gamma=gamma,
    kp=pid_kp, ki=pid_ki, kd=pid_kd,
    alpha=alpha,
    beta=beta,
    learning_rate=(lr_P, lr_I, lr_D),
)

# ─── 5) Run both agents for 20 000 steps & record value‐error ────────────
max_steps = 200000
n_runs = 1

hist_q   = np.zeros((n_runs, max_steps))
hist_pid = np.zeros((n_runs, max_steps))

kp_hist_q = np.zeros((n_runs, max_steps))
ki_hist_q = np.zeros((n_runs, max_steps))
kd_hist_q = np.zeros((n_runs, max_steps))
kp_hist_pid = np.zeros((n_runs, max_steps))
ki_hist_pid = np.zeros((n_runs, max_steps))
kd_hist_pid = np.zeros((n_runs, max_steps))

def make_value_error_test(env, agent=None, gamma=gamma):
    """
    Returns a test_fn(Q, Qp, BR) which computes
      ‖V_t - V*‖_1 / ‖V*‖_1
    for the deterministic “always-right” policy, where V* is found by
    synchronous policy‐evaluation on the true MDP.
    """
    # 1) pull out the full transition and reward arrays
    #    P[s, s2, a] = Pr(s→s2 | a),  R[s, a] = E[r | s, a]
    P_full = env.build_probability_transition_kernel()
    R_full = env.build_reward_matrix()
    nS, _, nA = P_full.shape

    # 2) build the “always-right” policy by choosing the action
    #    with highest immediate reward R[s,a] at each state
    optimal_pi = { s: int(np.argmax(R_full[s])) for s in range(nS) }

    # 3) synchronous policy-evaluation to find V*
    V = np.zeros(nS)
    while True:
        delta = 0.0
        for s in range(nS):
            a = optimal_pi[s]
            v_new = R_full[s, a] + gamma * np.dot(P_full[s, :, a], V)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < 1e-6:
            break

    V_norm = np.linalg.norm(V, ord=1)

    # 4) build the test function for your runs
    def test_fn(Q, Qp, BR):
        # record gains if desired
        if agent is not None:
            agent.kp_history.append(agent.kp)
            agent.ki_history.append(agent.ki)
            agent.kd_history.append(agent.kd)
        # compute current estimate V_t(s) = max_a Q(s,a)
        Vt = Q.max(axis=1)
        return np.linalg.norm(Vt - V, ord=1) / V_norm

    return test_fn


test_fn_q = make_value_error_test(env,agent=agent_q)
test_fn_pid = make_value_error_test(env,agent=agent_pid)

print("Running pure Q-Learning baseline…")
for run in range(n_runs):
    print(f"Run {run+1}/{n_runs} (Q-learning)…")
    # Reset the environment and agent for each run
    env.reset()
    agent_q.reset()
    agent_q.kp_history = []
    agent_q.ki_history = []
    agent_q.kd_history = []

    # Run the agents for max_steps
    h_q, _ = agent_q.estimate_value_function(
        num_iterations=max_steps,
        test_function=test_fn_q,
        stop_if_diverging=False,
        follow_trajectory=False,
        reset_environment=True
    )
    kp_hist_q[run, :] = agent_q.kp_history
    ki_hist_q[run, :] = agent_q.ki_history
    kd_hist_q[run, :] = agent_q.kd_history
    hist_q[run, :] = h_q

print("Running PID-Accelerated TD-Learning…")
for run in range(n_runs):
    print(f"Run {run+1}/{n_runs} (PID) …")
    # Reset the environment and agent for each run
    env.reset()
    agent_pid.reset()
    agent_pid.kp_history = []
    agent_pid.ki_history = []
    agent_pid.kd_history = []

    # Run the agents for max_steps
    h_pid, _ = agent_pid.estimate_value_function(
        num_iterations=max_steps,
        test_function=test_fn_pid,
        stop_if_diverging=False,
        follow_trajectory=False,
        reset_environment=True
    )
    kp_hist_pid[run, :] = agent_pid.kp_history
    ki_hist_pid[run, :] = agent_pid.ki_history
    kd_hist_pid[run, :] = agent_pid.kd_history
    hist_pid[run, :] = h_pid

# ─── 6) Plot the normalized ‖Vₜ−V*‖₁ curves ─────────────────────────────
fig, (ax_err, ax_kp, ax_ki, ax_kd) = plt.subplots(
    nrows=1, ncols=4,
    figsize=(20, 4),
    gridspec_kw={'width_ratios': [4, 1, 1, 1], 'wspace': 0.4}
)
ts = np.arange(max_steps)

# 1) main error curves
plotSEM(ts, hist_q,   label="TD-Learning", color="tab:blue",   ax=ax_err)
plotSEM(ts, hist_pid,  label="PID-Accelerated TD", color="tab:orange", ax=ax_err)
ax_err.set_xlabel("Steps (t)")
ax_err.set_ylabel(r"Normalized $\|V_t - V^*\|_1$")
ax_err.set_title("Value‐Error Over Time")
ax_err.legend(loc="upper right")
ax_err.grid(True)

# 2) Kp subplot
plotSEM(ts, kp_hist_q,   label="TD Kp",   color="tab:blue", ax=ax_kp)
plotSEM(ts, kp_hist_pid, label="PID Kp", color="tab:orange", ax=ax_kp)
ax_kp.set_title("Kp (Proportional Gain)")
ax_kp.set_xlabel("Steps")
ax_kp.set_yticks([0.0, 0.5, 1.0, 1.5])
ax_kp.legend()

# 3) Ki subplot
plotSEM(ts, ki_hist_q,   label="TD Ki",   color="tab:blue", ax=ax_ki)
plotSEM(ts, ki_hist_pid, label="PID Ki", color="tab:orange", ax=ax_ki)
ax_ki.set_title("Ki (Integral Gain)")
ax_ki.set_xlabel("Steps")
ax_ki.set_yticks([0.0, 0.5, 1.0])
ax_ki.legend()

# 4) Kd subplot
plotSEM(ts, kd_hist_q,   label="TD Kd",   color="tab:blue", ax=ax_kd)
plotSEM(ts, kd_hist_pid, label="PID Kd", color="tab:orange", ax=ax_kd)
ax_kd.set_title("Kd (Derivative Gain)")
ax_kd.set_xlabel("Steps")
ax_kd.set_yticks([0.0, 0.5, 1.0])
ax_kd.legend()

plt.tight_layout()
plt.show()