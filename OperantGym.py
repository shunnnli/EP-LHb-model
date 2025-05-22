import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class OperantLearning(gym.Env):
    """
    Environment for a cue-lick operant task:
      - ENL: withhold licking for 2-4s (100ms steps) to start trial
      - Tone/response: animal has 500ms cue + 2s response window
      - Big outcome delivered immediately upon 2+ licks
      - Small outcome delivered at end of 2s if <2 licks
      - Omission trials: no outcome (reward/punishment) delivered based on omission_prob
    """
    def __init__(self, pairing='reward', omission_prob: float = 0.0, 
                 enl_duration: tuple[float, float] = (2.0, 4.0),
                 action_cost: float = 0.1, enl_penalty: float = 0.01,
                 detection_delay: int = 0,):
        super().__init__()
        # Actions: 0 = no lick, 1 = lick
        self.action_space = spaces.Discrete(2)
        # Observations: [phase, time_in_phase, cue_on]
        # self.max_time = int(enl_duration[1] * 10) + 20  # max ENL time + response window
        # self.observation_space = spaces.MultiDiscrete([2, self.max_time, 2])
        self.observation_space = spaces.MultiDiscrete([2, 2])

        # Reward structures
        self.enl_penalty = enl_penalty
        self.action_cost = action_cost

        # Trial parameters
        self.omission_prob = omission_prob
        self.trial_type = pairing  # "reward" or "punish"
        self.enl_duration_range = (int(enl_duration[0] * 10), int(enl_duration[1] * 10))  # [min, max] in seconds
        
        # how many steps to delay reward detection
        self.detection_delay = detection_delay + 1  # +1 to account for the first step
        self._reward_buffer = deque([0]*self.detection_delay, maxlen=self.detection_delay)
        self._pending_reset_steps = 0

        # Internal state
        self.phase = 0             # 0 = ENL, 1 = response
        self.time = 0
        self.enl_duration = 0
        self.lick_buffer = []
        self.cue_on = 0
        self.omission_trial = False
        self.last_trial_info = None
        self.outcome_type = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.phase = 0
        self.time = 0
        self.enl_duration = np.random.randint(self.enl_duration_range[0], self.enl_duration_range[1])  # 2-4s in 100ms steps
        self.lick_buffer = []
        self.cue_on = 0
        self.omission_trial = (np.random.rand() < self.omission_prob)
        self.last_trial_info = None
        self.outcome_type = None
        # clear reward buffer
        self._pending_reset_steps = 0
        self._reward_buffer.clear()
        self._reward_buffer.extend([0]*self.detection_delay)
        return self._get_obs(), {}

    def _get_obs(self):
        # noisy_time = int(self.time + np.random.randn()*10)
        # noisy_time = np.clip(noisy_time, 0, self.max_time)
        return np.array([self.phase, self.cue_on], dtype=int)

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Incur action cost for licking
        if action == 1:
            reward -= self.action_cost

        if self.phase == 0:
            # ENL period: reset if lick, count if no lick
            if action == 1:
                self.time = 0
                self.enl_duration = np.random.randint(self.enl_duration_range[0], self.enl_duration_range[1])
                reward -= self.enl_penalty
                info = {
                    "lick": 1,
                    "reward": reward,
                    "done": False,
                    "outcome": "enl_break",
                }
            else:
                self.time += 1
                if self.time >= self.enl_duration:
                    # transition to response window
                    self.phase = 1
                    self.time = 0
                    self.lick_buffer = []
                    self.cue_on = 1
                    print("     Cue ON")
                    
        else:
            # Response phase: collect licks and time
            if action == 1: self.lick_buffer.append(action)
            self.time += 1
            
            # Turn off cue after 500ms (5 steps)
            if self.time >= 5:
                self.cue_on = 0

                # Check immediate big outcome
                if len(self.lick_buffer) >= 2 and action == 1 and self.outcome_type is None:
                    self.outcome_type = "big"
                    raw_outcome = 10 if self.trial_type == "reward" else -10
                    # apply omission
                    outcome = 0 if self.omission_trial else raw_outcome
                    reward += outcome
                    info = {
                        "lick": len(self.lick_buffer),
                        "reward": reward,
                        "done": False,
                        "outcome": "omission" if self.omission_trial else self.outcome_type,
                    }
                    # schedule trial reset after delay
                    if self.detection_delay > 0:
                        self._pending_reset_steps = self.detection_delay
                    else:
                        self._reset_trial()
                    print("     Big outcome delivered")

                # Check end of response window for small outcome
                elif self.time >= 20 and self.outcome_type is None:
                    self.outcome_type = "small"
                    raw_outcome = 2 if self.trial_type == "reward" else -2
                    outcome = 0 if self.omission_trial else raw_outcome
                    reward += outcome
                    info = {
                        "lick": len(self.lick_buffer),
                        "reward": reward,
                        "done": False,
                        "outcome": "omission" if self.omission_trial else self.outcome_type,
                    }
                    # schedule trial reset after delay
                    if self.detection_delay > 0:
                        self._pending_reset_steps = self.detection_delay
                    else:
                        self._reset_trial()
                    print("     Small outcome delivered")

        # Fill placeholders if no outcome yet
        if not info:
            info = {
                "lick": len(self.lick_buffer),
                "reward": 0,
                "done": False,
                "outcome": self.outcome_type,
            }
        
        # implement detection delay: buffer raw_reward before returning
        if self.detection_delay > 0:
            # buffer raw reward
            self._reward_buffer.append(reward)
            # pop oldest (which occurred detection_delay steps ago)
            final_reward = self._reward_buffer.popleft()
            # after delivering the delayed outcome, reset the trial
            if self._pending_reset_steps > 0:
                self._pending_reset_steps -= 1
                if self._pending_reset_steps == 0:
                    info = {
                        "lick": len(self.lick_buffer),
                        "reward": reward,
                        "done": True,
                        "outcome": "trial_end",
                    }
                    self._reset_trial()
        else:
            final_reward = reward

        return self._get_obs(), final_reward, terminated, truncated, info

    def _reset_trial(self):
        """Reset internal state for next trial after outcome delivery."""
        self.phase = 0
        self.time = 0
        self.enl_duration = np.random.randint(20, 40)
        self.lick_buffer = []
        self.cue_on = 0
        self.omission_trial = (np.random.rand() < self.omission_prob)
        self.outcome_type = None
        # clear reward buffer
        self._pending_reset_steps = 0
        self._reward_buffer.clear()
        self._reward_buffer.extend([0]*self.detection_delay)
