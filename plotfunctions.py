import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
    
def get_traces(data, event, pre_steps, post_steps):
    """
    Aligns data to event indices.
    """
    data = np.asarray(data)

    # Extract rising edge if necessary
    if len(data) == len(event):
        event_idx = np.where(np.diff(event) == 1)[0] + 1
    else:
        event_idx = event

    n_trials = len(event_idx)
    window_len = pre_steps + post_steps
    T = len(data)

    aligned_data = np.zeros((n_trials, window_len), dtype=data.dtype)

    for i, idx in enumerate(event_idx):
        start = idx - pre_steps + 1
        end   = idx + post_steps
        # fill in the valid slice
        lo = max(start, 0)
        hi = min(end, T-1)
        # these are the positions within the window array
        w_lo = lo - start
        w_hi = w_lo + (hi - lo) + 1

        aligned_data[i, w_lo:w_hi] = data[lo : hi+1]

    return aligned_data
    
def plot_figure(licks, tds, reward_history, loss_history,
                dt=0.1,
                pre_steps=20, post_steps=30,
                tau_on: float = 0.01, tau_off: float = 0.1):
    
    # convert to numpy arrays if not already
    if not isinstance(licks, np.ndarray):
        licks = np.array(licks)
    if not isinstance(tds, np.ndarray):
        tds = np.array(tds)

    # time axis from -1s to +2s at 0.1s steps
    dt = 0.1
    max_trial_steps = pre_steps + post_steps
    t_axis = np.linspace(-pre_steps*dt, (max_trial_steps-pre_steps)*dt, max_trial_steps)

    # number of trials
    num_trials = licks.shape[0]

    # build DA kernel
    t_kernel = np.arange(0, 1.0, dt)  # 0–1 s
    kernel = np.exp(-t_kernel/tau_off) - np.exp(-t_kernel/tau_on)
    kernel /= np.sum(kernel)  # normalize area to 1
    # convolve with DA kernel
    DAs = np.stack([
        np.convolve(trial_tds, kernel, mode='full')[:tds.shape[1]]
        for trial_tds in tds
    ], axis=0)

    # Plotting
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.2, hspace=0.3)

    # top‐left: Reward per trial
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(reward_history)
    ax0.set_title("Reward per trial")
    ax0.set_xlabel("Trial")
    ax0.set_ylabel("Total Reward")

    # top‐middle: Loss per trial
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(loss_history)
    ax1.set_title("Loss per trial")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("MSE Loss")

    # bottom‐left: Lick raster (scatter)
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(num_trials):
        lick_times = t_axis[licks[i] == 1]
        ax2.scatter(lick_times, np.ones_like(lick_times)*(i+1),
                    color='tab:pink', s=20, marker='o', alpha=0.8, edgecolor='none')
    # mark cue window
    ax2.fill_betweenx([0, num_trials+1], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
    ax2.set_title("Lick raster")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Trial")
    ax2.set_xlim(t_axis[0], t_axis[-1])
    ax2.set_ylim(0.5, num_trials+1)

    # bottom‐middle: average TD error + simulated DA signal
    ax3 = fig.add_subplot(gs[1, 1])
    plotSEM(t_axis, tds, label="TD Error", color='tab:blue', ax=ax3, alpha=0.2)
    plotSEM(t_axis, DAs, label="DA Signal", color='tab:green', ax=ax3, alpha=0.2)
    # shade cue
    ymin, ymax = ax3.get_ylim()
    ax3.fill_betweenx([ymin, ymax], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
    ax3.set_ylim(ymin, ymax)
    ax3.set_title("TD error vs time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("TD Error")
    ax3.legend(loc='upper left', fontsize=10, frameon=False)

    # the new heatmap, spanning both rows in column 3
    ax_heat = fig.add_subplot(gs[:, 2])
    data = tds
    im = ax_heat.imshow(
        data,
        aspect='auto',
        interpolation='nearest',
        extent=[t_axis[0], t_axis[-1], num_trials, 1],
        cmap='RdBu_r',
        vmin=-np.max(np.abs(data)),
        vmax=np.max(np.abs(data)),
    )
    ax_heat.set_title("Heatmap")
    ax_heat.set_xlabel("Time (s)")
    ax_heat.set_ylabel("Trial")
    ax_heat.invert_yaxis()   # so Trial 1 is at top
    cbar = fig.colorbar(im, ax=ax_heat, orientation='vertical', label='TD Error')

    # Tighten layout and show
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.07)
    plt.show()