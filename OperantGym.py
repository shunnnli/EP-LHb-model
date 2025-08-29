import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class OperantLearning(gym.Env):
    """
    Environment for a cue-lick operant task with hand-designed curriculum:
      - ENL: withhold licking for 2-4s (100ms steps) to start trial
      - Tone/response: animal has 500ms cue + 2s response window
      - Big outcome delivered immediately upon 2+ licks
      - Small outcome delivered at end of 2s if <2 licks
      - Omission trials based on omission_prob
      - Difficulty (level) increases when agent >80% success over last 100 trials
        * On level-up, omission_prob increases by 0.1 (capped at 1.0)
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 30,
    }

    def __init__(self,
                 pairing='reward',
                 omission_prob: float = 0.0,
                 enl_duration: tuple[float, float] = (2.0, 4.0),
                 action_cost: float = 0.1,
                 enl_penalty: float = 0.01,
                 detection_delay: int = 0,
                 render_mode: str = None,
                 continual_learning: bool = False, change_start: int = 200,
                 change_interval: int = 50,
                 printing: bool = False):
        super().__init__()
        # action/observation
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete([2, 2])

        # rewards
        self.action_cost = action_cost
        self.enl_penalty = enl_penalty

        # base params
        self.base_omission = omission_prob
        self.omission_prob = omission_prob
        self.trial_type = pairing
        self.enl_duration_range = (int(enl_duration[0]*10), int(enl_duration[1]*10))
        self.min_lick = 2

        # detection delay
        self.detection_delay = detection_delay + 1
        self._reward_buffer = deque([0]*self.detection_delay, maxlen=self.detection_delay)
        self._pending_reset_steps = 0

        # continual learning
        self.continual_learning = continual_learning
        self.change_start = change_start
        self.change_interval = change_interval
        self.level = 0
        self.prev_level = 0
        self.swap_rewards = False # False means default, True means big = 2, small = 10

        # state
        self.phase = 0
        self.time = 0
        self.enl_duration = 0
        self.lick_buffer = []
        self.cue_on = 0
        self.trial_count = 0
        self.omission_trial = False
        self.outcome_type = None

        # render
        self.render_mode = render_mode
        self._screen = None
        self.printing = printing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # sample ENL for this trial from fixed range
        self.phase = 0
        self.time = 0
        self._apply_difficulty()             # updates omission_prob based on level
        self.enl_duration = np.random.randint(*self.enl_duration_range)
        self.lick_buffer = []
        self.cue_on = 0
        self.outcome_type = None
        self.omission_trial = (np.random.rand() < self.omission_prob)
        self._pending_reset_steps = 0
        self._reward_buffer.clear()
        self._reward_buffer.extend([0]*self.detection_delay)
        return self._get_obs(), {}

    def _apply_difficulty(self):
        # on level-up, omission_prob increases by 0.1, cap at 1.0
        # if self.level > 8 and self.level % 2 == 1:
        #     level = 0
        # else:
        #     level = self.level
        # self.omission_prob = min(self.base_omission + 0.1 * level, 0.9)
        

        if self.level != self.prev_level:
            self.prev_level = self.level
            # self.omission_prob = np.random.choice(np.arange(0.0, 0.9, 0.1))
            # print("self.omission prob:", self.omission_prob)
            # Swap rewards every level-up (or every other, if you prefer)
            self.swap_rewards = not self.swap_rewards
        
        # if self.level != self.prev_level:
        #     self.prev_level = self.level
        #     if self.omission_prob == 0.2 or self.omission_prob == 0:
        #         self.omission_prob = 0.8
        #         print("omisssion prob:", self.omission_prob)
        #     elif self.omission_prob == 0.8:
        #         self.omission_prob = 0.2
        #         print("omisssion prob:", self.omission_prob)

    def _get_obs(self):
        return np.array([self.phase, self.cue_on], dtype=int)

    def _reset_trial(self):
        # basic reset
        self.phase = 0
        self.time = 0
        if self.continual_learning:
            self._apply_difficulty()           # update omission_prob if level changed
        self.enl_duration = np.random.randint(*self.enl_duration_range)
        self.lick_buffer = []
        self.cue_on = 0
        self.outcome_type = None
        self.omission_trial = (np.random.rand() < self.omission_prob)
        self._pending_reset_steps = 0
        self._reward_buffer.clear()
        self._reward_buffer.extend([0]*self.detection_delay)

    def step(self, action):
        reward = 0
        info = {}
        # cost
        if action == 1:
            reward -= self.action_cost

        # ENL phase
        if self.phase == 0:
            if action == 1:
                # ENL break
                self.time = 0
                reward -= self.enl_penalty
                info = {"lick": 1, "cue": self.cue_on, "done": True, "outcome": "enl_break"}
                self._reset_trial()
            else:
                self.time += 1
                if self.time >= self.enl_duration:
                    # go to response
                    self.phase = 1
                    self.time = 0
                    self.lick_buffer = []
                    self.cue_on = 1
                    info = {"lick": 0, "cue": self.cue_on, "done": False, "outcome": "trial_start"}
                else:
                    info = {"lick": len(self.lick_buffer), "cue": self.cue_on, "done": False, "outcome": "enl_ongoing"}
        else:
            # Response phase
            if action == 1:
                self.lick_buffer.append(action)
            self.time += 1
            if self.time >= 5:
                self.cue_on = 0
                # Big outcome
                if len(self.lick_buffer) >= self.min_lick and action == 1 and self.outcome_type is None:
                    self.outcome_type = "big"
                    if self.swap_rewards:
                        raw = 2 if self.trial_type == "reward" else -2   # swapped
                    else:
                        raw = 10 if self.trial_type == "reward" else -10

                    out = 0 if self.omission_trial else raw
                    reward += out
                    info = {"lick": len(self.lick_buffer), "cue": self.cue_on,
                            "done": False, "outcome": ("omission" if self.omission_trial else "big")}
                    self._pending_reset_steps = self.detection_delay if self.detection_delay > 0 else 0
                    if self.detection_delay == 0:
                        self._reset_trial()

                # Small outcome
                elif self.time >= 20 and self.outcome_type is None:
                    self.outcome_type = "small"
                    if self.swap_rewards:
                        raw = 10 if self.trial_type == "reward" else -10   # swapped
                    else:
                        raw = 2 if self.trial_type == "reward" else -2

                    out = 0 if self.omission_trial else raw
                    reward += out
                    info = {"lick": len(self.lick_buffer), "cue": self.cue_on,
                            "done": False, "outcome": ("omission" if self.omission_trial else "small")}
                    self._pending_reset_steps = self.detection_delay if self.detection_delay > 0 else 0
                    if self.detection_delay == 0:
                        self._reset_trial()
                else:
                    info = {"lick": len(self.lick_buffer), "cue": self.cue_on, "done": False, "outcome": "cue_on"}         

        # defaults
        if not info:
            info = {"lick": len(self.lick_buffer), "cue": self.cue_on, "done": False, "outcome": self.outcome_type}

        # Delay delivery
        if self.detection_delay > 0 and info["outcome"] != "enl_break":
            self._reward_buffer.append(reward)
            final_reward = self._reward_buffer.popleft()
            if self._pending_reset_steps > 0:
                self._pending_reset_steps -= 1
                if self._pending_reset_steps == 0:
                    info = {"lick": len(self.lick_buffer), "cue": self.cue_on, "done": True, "outcome": "trial_end"}
                    self._reset_trial()
        else:
            final_reward = reward

        # Curriculum update at true trial end
        if info.get("done", False) and info.get("outcome") == "trial_end":
            level_up = False
            if self.continual_learning and self.trial_count % self.change_interval == 0 and self.trial_count >= self.change_start:
                self.level += 1
                level_up = True
            info["level_up"] = level_up
        else:
            info["level_up"] = False

        return self._get_obs(), final_reward, False, False, info