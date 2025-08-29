# List of Experiments

## 1. Basic Sweep of kd and omission probs, gain adapt on

```python
# kd_values        = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # PID derivative gain values
# meta_lr_d        = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Gain Adapt ON (0.1 meta_lr_d) for kd > 0, OFF (0) for kd = 0
# omission_probs   = [0, 0.1, 0.2, 0.3, 0.4, 0.5] # constant omission_prob throughout task
# repeats          = 50  # Number of repeats for each combination
```

## 2. Basic Sweep of kd and omission probs, gain adapt off

```python
# kd_values        = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # PID derivative gain values
# meta_lr_d        = [0, 0,   0,   0,   0,   0,   0,   0,   0,   0. ]  # Gain Adapt ON (0.1 meta_lr_d) for kd > 0, OFF (0) for kd = 0
# omission_probs   = [0, 0.1, 0.2, 0.3, 0.4, 0.5] # constant omission_prob throughout task
# repeats          = 50  # Number of repeats for each combination
```

## 3. Repeat Above for replay buffer size combinations

### 3. a. Buffer size of 10, 5 recent, 5 random old trials

```python
# "max_batch_size":   10,   # max replay buffer space
# "num_recent":        5,    # number of consecutive recent trials to fill replay buffer. ex. 5 num_recent, means 5 random old trials in size 10 replay buffer
```

### 3. b. Buffer size of 20, 5 recent, 15 random old trials

```python
# "max_batch_size":   20,
# "num_recent":        5,
```
