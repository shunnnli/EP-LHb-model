import numpy as np
import matplotlib.pyplot as plt

# Time axis
T = np.arange(0, 10, 0.01)  # 0 to 10 seconds at 100 Hz
dt = T[1] - T[0]

# Define patterns for V(t)
patterns = {
    'Triangular ramp': np.where((T >= 1) & (T <= 9),
                                np.where(T <= 5, (T-1)/4, (9-T)/4),
                                0.0),
    'Step pulse': np.where((T >= 1) & (T <= 9), 1.0, 0.0),
    'Sawtooth wave': np.where((T >= 1) & (T <= 9),
                               ((T-1) * 0.5) % 1,
                               0.0),
    'Exponential decay': np.where((T >= 1) & (T <= 9),
                                  np.exp(-(T-1)/2),
                                  0.0),
    'Gaussian bump': np.exp(-((T-5)**2)/(2*1.5**2)),
    'Biphasic pulse': (
         np.where((T>2)&(T<4),  1.0, 0) 
       - np.where((T>6)&(T<8),  1.0, 0)
    ),
    'Burst oscillations': np.where((T >= 4) & (T <= 6),
                   0.5 + 0.5 * np.sin(2 * np.pi * 5 * (T - 4)), 0.0)
}

gamma = 0.9  # discount factor
r = np.zeros_like(T)  # zero reward for simplicity

# Create subplots: 4 rows, 4 cols (V, δ, derivative dδ, TD of δ)
fig, axes = plt.subplots(len(patterns), 4, figsize=(16, 12), sharex='col')
fig.tight_layout(pad=4.0)

for row, (name, V) in enumerate(patterns.items()):
    # Compute TD error δ(t)
    delta = gamma * V[1:] + r[:-1] - V[:-1]
    # Compute derivative (finite difference) of δ: dδ/dt
    deriv_delta = np.diff(delta) / dt
    # Compute TD of δ: δ2(t) = γ δ(t+1) - δ(t)
    delta2 = gamma * delta[1:] + r[:-2] - delta[:-1]
    
    # Plot V(t)
    axes[row, 0].plot(T, V)
    axes[row, 0].set_ylabel(name, rotation=90, va='center')
    axes[row, 0].set_title('V(t)')
    axes[row, 0].grid(True)
    
    # Plot δ(t)
    axes[row, 1].plot(T[:-1], delta)
    axes[row, 1].set_title('TD')
    axes[row, 1].grid(True)
    
    # Plot derivative of δ
    axes[row, 2].plot(T[:-2], deriv_delta)
    axes[row, 2].set_title('dTD/dt')
    axes[row, 2].grid(True)
    
    # Plot TD of δ
    axes[row, 3].plot(T[:-2], delta2)
    axes[row, 3].set_title('TD of TD')
    axes[row, 3].grid(True)

# Set common x-label
for ax in axes[-1, :]:
    ax.set_xlabel('t (s)')

plt.suptitle(f'Summary of patterns (gamma = {gamma})', y=1.02, fontsize=16)
plt.subplots_adjust(top=0.93)
plt.show()
