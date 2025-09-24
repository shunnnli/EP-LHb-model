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

    # Fake rendering mode for compatibility with SB3
    # (not actually rendering anything)
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 30,
    }

    def __init__(self, pairing='reward', omission_prob: float = 0.0, 
                 enl_duration: tuple[float, float] = (2.0, 4.0),
                 trial_start: str = 'enl_start', detection_delay: int = 1,
                 action_cost: float = 0.1, enl_penalty: float = 0.1,
                 reward_decay: bool = True, reward_decay_time: float = 1.0,
                 render_mode: str = None,
                 continual_learning: bool = False, change_start: int = 200,
                 change_interval: int = 50,
                 print_status: bool = False):
        super().__init__()

        # Actions: 0 = no lick, 1 = lick
        self.action_space = spaces.Discrete(2)
        
        if trial_start == 'enl_start':
            # Observations: [phase, cue_on, licks_since_reward]
            # self.observation_space = spaces.Box(
            #     low=np.array([0, 0, 0]),  # phase is binary, cue_on is binary, licks_since_reward is non-negative
            #     high=np.array([1, 1, 100]),   # reasonable bounds for phase, cue_on is binary, licks_since_reward has reasonable upper bound
            #     dtype=np.float32
            # )
            self.observation_space = spaces.MultiDiscrete([2, 2])
        elif trial_start == 'cue_start':
            # Observations: [delivered_reward, cue_on, licks_since_reward]
            self.observation_space = spaces.Box(
                low=np.array([-20.0, 0, 0]),  # delivered_reward can be negative, cue_on is binary, licks_since_reward is non-negative
                high=np.array([20.0, 1, 100]),   # reasonable bounds for reward, cue_on is binary, licks_since_reward has reasonable upper bound
                dtype=np.float32
            )

        # Reward structures
        self.enl_penalty = enl_penalty
        self.action_cost = action_cost
        self.trial_start = trial_start

        # how many steps to delay reward detection
        self.detection_delay = detection_delay + 1  # +1 to account for the first step
        self._reward_buffer = deque([0]*self.detection_delay, maxlen=self.detection_delay)
        self._pending_reset_steps = 0

        # Trial parameters
        self.omission_prob = omission_prob
        self.trial_type = pairing  # "reward" or "punish"
        self.enl_duration_range = (int(enl_duration[0] * 10), int(enl_duration[1] * 10))  # [min, max] in seconds
        
        # Reward decay parameters
        self.reward_decay = reward_decay  # whether to use decay mechanism
        self.reward_decay_time = reward_decay_time  # time in seconds for decay
        self.reward_decay_steps = int(reward_decay_time * 10)  # convert to timesteps
        # Calculate decay factor to reach 0.01 after reward_decay_steps
        self.reward_decay_factor = 0.01 ** (1.0 / self.reward_decay_steps) if self.reward_decay_steps > 0 else 1.0

        # Internal state
        self.time = 0
        self.phase = 0  # 0 = ENL, 1 = response
        self.enl_duration = 0
        self.lick_buffer = []
        self.cue_on = 0
        self.omission_trial = False
        self.last_trial_info = None
        self.outcome_type = None
        self.pending_reward = 0  # reward waiting to be delivered on next lick
        self.reward_start_time = 0  # when the reward became available
        self.enl_start_time = 0  # when ENL period started
        self.cur_enl_duration = 0  # whether currently in ENL period
        self.licks_since_reward = 0  # number of licks since reward became available
        self.is_reward = False  # whether any reward has been delivered in current trial
        self.delivered_reward = 0.0  # current delivered reward for observation

        # continual learning parameters
        self.level = 0
        self.prev_level = 0
        self.swap_rewards = False   # False = normal rewards, True = swapped
        self.change_start = change_start  # start changing levels after this many trials
        self.change_interval = change_interval
        self.continual_learning = continual_learning  # whether to enable continual learning
        self.trial_count = 0

        # Fake render mode
        self.render_mode = render_mode
        self._screen = None

        self.print_status = print_status  # whether to print debug info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.phase = 0
        self._apply_difficulty()
        self.enl_duration = np.random.randint(self.enl_duration_range[0], self.enl_duration_range[1])  # 2-4s in 100ms steps
        self.lick_buffer = []
        self.cue_on = 0
        self.omission_trial = (np.random.rand() < self.omission_prob)
        self.last_trial_info = None
        self.outcome_type = None
        self.pending_reward = 0
        self.reward_start_time = 0
        self.enl_start_time = 0
        self.cur_enl_duration = 0
        self.licks_since_reward = 0
        self.is_reward = False
        self.delivered_reward = 0.0

        # clear reward buffer
        self._pending_reset_steps = 0
        self._reward_buffer.clear()
        self._reward_buffer.extend([0]*self.detection_delay)

        return self._get_obs(), {}

    def _get_obs(self):
        # Return delivered reward and cue status
        if self.trial_start == 'enl_start':
            return np.array([self.phase, self.cue_on], dtype=np.float32)
        elif self.trial_start == 'cue_start':
            # Return delivered reward, cue status, and licks since last reward
            return np.array([self.delivered_reward, self.cue_on, self.licks_since_reward], dtype=np.float32)
        
    def _apply_difficulty(self):
        """
        Adjust environment parameters whenever level increases.
        Currently:
          - toggles reward swapping
          - (optional) adjust omission_prob or other difficulty factors
        """
        if self.level != self.prev_level:
            self.prev_level = self.level
            # Toggle reward swap
            self.swap_rewards = not self.swap_rewards
            if self.print_status:
                print(f"Level-up! New level = {self.level}, swapped rewards = {self.swap_rewards}")

    def _reset_trial(self):
        """Reset internal state for next trial after outcome delivery."""
        self.time = 0
        self.phase = 0
        self._apply_difficulty()
        self.enl_duration = np.random.randint(20, 40)
        self.lick_buffer = []
        self.cue_on = 0
        self.omission_trial = (np.random.rand() < self.omission_prob)
        self.outcome_type = None
        self.pending_reward = 0
        self.reward_start_time = 0
        self.enl_start_time = 0
        self.cur_enl_duration = 0
        self.licks_since_reward = 0
        self.is_reward = False
        self.delivered_reward = 0.0

        # clear reward buffer
        self._pending_reset_steps = 0
        self._reward_buffer.clear()
        self._reward_buffer.extend([0]*self.detection_delay)

    def render(self, mode='human'):
        if self.render_mode == "rgb_array":
            # return a dummy image or your real frame
            if self._screen is None:
                h, w = 400, 400
                self._screen = np.zeros((h, w, 3), dtype=np.uint8)
            return self._screen
        elif self.render_mode == "human":
            print("Rendering (human mode)")
        else:
            return None

    def step(self, action):
        if self.trial_start == 'enl_start':
            return self.step_start_with_enl(action)
        elif self.trial_start == 'cue_start':
            return self.step_start_with_cue(action)
        else:
            # Default to cue_start if trial_type is not recognized
            return self.step_start_with_enl(action)


    def step_start_with_enl(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}
        print_status = self.print_status

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
                    "cue": self.cue_on,
                    "done": True,
                    "outcome": "enl_break",
                }
                self._reset_trial()
            else:
                self.time += 1
                if self.time >= self.enl_duration:
                    # transition to response window
                    self.phase = 1
                    self.time = 0
                    self.lick_buffer = []
                    self.cue_on = 1
                    info = {
                        "lick": 0,
                        "cue": self.cue_on,
                        "done": False,
                        "outcome": "trial_start",
                    }
                    if print_status: print(f"     Time {self.time}: Cue ON")
                else:
                    info = {
                        "lick": len(self.lick_buffer),
                        "cue": self.cue_on,
                        "done": False,
                        "outcome": "enl_ongoing",
                    }
                    
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
                    if self.swap_rewards:
                        raw_outcome = 2 if self.trial_type == "reward" else -2
                    else:
                        raw_outcome = 10 if self.trial_type == "reward" else -10

                    # apply omission
                    outcome = 0 if self.omission_trial else raw_outcome
                    reward += outcome
                    info = {
                        "lick": len(self.lick_buffer),
                        "cue": self.cue_on,
                        "done": False,
                        "outcome": "omission" if self.omission_trial else self.outcome_type,
                    }
                    # schedule trial reset after delay
                    if self.detection_delay > 0:
                        self._pending_reset_steps = self.detection_delay
                    else:
                        self._reset_trial()
                    if print_status: print(f"     Time {self.time}: Big outcome delivered")

                # Check end of response window for small outcome
                elif self.time >= 20 and self.outcome_type is None:
                    self.outcome_type = "small"
                    if self.swap_rewards:
                        raw_outcome = 10 if self.trial_type == "reward" else -10
                    else:
                        raw_outcome = 2 if self.trial_type == "reward" else -2

                    outcome = 0 if self.omission_trial else raw_outcome
                    reward += outcome
                    info = {
                        "lick": len(self.lick_buffer),
                        "cue": self.cue_on,
                        "done": False,
                        "outcome": "omission" if self.omission_trial else self.outcome_type,
                    }
                    # schedule trial reset after delay
                    if self.detection_delay > 0:
                        self._pending_reset_steps = self.detection_delay
                    else:
                        self._reset_trial()
                    if print_status: print(f"     Time {self.time}: Small outcome delivered")

            else:
                info = {
                    "lick": len(self.lick_buffer),
                    "cue": self.cue_on,
                    "done": False,
                    "outcome": "cue_on",
                }

        # Fill placeholders if no outcome yet
        if not info:
            info = {
                "lick": len(self.lick_buffer),
                "cue": self.cue_on,
                "done": False,
                "outcome": self.outcome_type,
            }
        
        # implement detection delay: buffer raw_reward before returning
        if self.detection_delay > 0 and info["outcome"] != "enl_break":
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
                        "cue": self.cue_on,
                        "done": True,
                        "outcome": "trial_end",
                    }
                    self._reset_trial()
        else:
            final_reward = reward
        
        # === After trial finishes ===
        if info.get("done", False) and info.get("outcome") == "trial_end":
            self.trial_count += 1
            level_up = False
            # simple rule: every 50 trials after 200, level up
            if self.continual_learning and self.trial_count % self.change_interval == 0 and self.trial_count >= self.change_start:
                self.level += 1
                level_up = True
            info["level_up"] = level_up
        else:
            info["level_up"] = False

        return self._get_obs(), final_reward, terminated, truncated, info


    def step_start_with_cue(self, action):
        delivered_reward = 0  # Initialize delivered_reward - this is what agent sees
        cur_reward = 0
        terminated = False
        truncated = False
        info = {}
        print_status = self.print_status

        # Incur action cost for licking
        # Continuous ENL check throughout the trial
        if action == 1:
            # Reset ENL period when agent licks
            self.enl_start_time = self.time
            self.cur_enl_duration = 0
            # incur action cost and ENL penalty 
            delivered_reward -= self.action_cost
            if self.licks_since_reward > 10:
                self.enl_penalty += self.enl_penalty
                delivered_reward -= self.enl_penalty
        else: 
            self.cur_enl_duration += 1
        
        # Start trial with cue ON
        if self.time == 0:
            self.cue_on = 1
            self.phase = 1
            if print_status: 
                print(f"     Time {self.time}: Cue ON (ENL duration: {self.enl_duration})")
        
        # Turn off cue after 500ms (5 steps)
        if self.time >= 5:
            self.cue_on = 0
        
        # End trial if ENL period is complete
        if self.cur_enl_duration >= self.enl_duration:
            # ENL period complete: end trial
            if print_status: print(f"     Time {self.time}: trial end, ENL complete")
            info = {
                "lick": len(self.lick_buffer),
                "cue": self.cue_on,
                "done": True,
                "outcome": "trial_end",
            }
            self._reset_trial()
            return self._get_obs(), delivered_reward, terminated, truncated, info

        # End trial if 100 sec have passed
        if self.time >= 999:
            # ENL period complete: end trial
            if print_status: print(f"     Time {self.time}: trial end, max time reached")
            info = {
                "lick": len(self.lick_buffer),
                "cue": self.cue_on,
                "done": True,
                "outcome": "trial_end",
            }
            self._reset_trial()
            return self._get_obs(), delivered_reward, terminated, truncated, info
        
        # Handle licking and reward collection
        if action == 1:
            self.lick_buffer.append(action)
            
            # Check for big outcome (2+ licks) after cue is off
            if self.cue_on == 0 and len(self.lick_buffer) >= 2 and self.time < 20 and self.outcome_type is None:
                self.outcome_type = "big"
                if self.swap_rewards:
                        raw_outcome = 2 if self.trial_type == "reward" else -2
                else:
                    raw_outcome = 10 if self.trial_type == "reward" else -10
                outcome = 0 if self.omission_trial else raw_outcome
                self.phase = 0
                
                if outcome != 0:
                    self.pending_reward = outcome
                    self.reward_start_time = self.time
                    self.licks_since_reward = 0  # Reset lick counter for new reward
                    if print_status: print(f"     Time {self.time}: Big outcome available: {outcome}")
                
                info = {
                    "lick": len(self.lick_buffer),
                    "cue": self.cue_on,
                    "done": False,
                    "outcome": "omission" if self.omission_trial else self.outcome_type,
                }
            
            # Check for small outcome at end of trial (20 steps = 2 seconds)
            elif self.time >= 20 and self.outcome_type is None:
                self.outcome_type = "small"
                if self.swap_rewards:
                        raw_outcome = 10 if self.trial_type == "reward" else -10
                else:
                    raw_outcome = 2 if self.trial_type == "reward" else -2
                outcome = 0 if self.omission_trial else raw_outcome
                self.phase = 0
                
                if outcome != 0:
                    self.pending_reward = outcome
                    self.reward_start_time = self.time
                    self.licks_since_reward = 0  # Reset lick counter for new reward
                    if print_status: print(f"     Time {self.time}: Small outcome available: {outcome}")
                
                info = {
                    "lick": len(self.lick_buffer),
                    "cue": self.cue_on,
                    "done": False,
                    "outcome": "omission" if self.omission_trial else self.outcome_type,
                }
        
            # Check if there's pending reward to deliver (only after cue is off)
            if self.cue_on == 0 and self.pending_reward > 0:
                # Mark that a reward has been delivered in this trial
                self.is_reward = True

                if self.reward_decay:
                    # Use decay mechanism
                    decay_factor = self.reward_decay_factor ** self.licks_since_reward  # decay per lick
                    cur_reward = self.pending_reward * decay_factor
                    delivered_reward += cur_reward
                    if print_status: print(f"     Time {self.time}: Lick (obtained {cur_reward:.2f} reward)")

                else:
                    # Give all reward at first lick
                    cur_reward = self.pending_reward
                    delivered_reward += cur_reward
                    if print_status: print(f"     Time {self.time}: Lick (obtained {cur_reward:.2f} reward)")
                    # Clear pending reward after first delivery
                    self.pending_reward = 0

            # Print and update lick info
            if print_status and not self.is_reward: print(f"     Time {self.time}: Lick")
            if self.is_reward: self.licks_since_reward += 1

        # Fill placeholders if no info set
        if not info:
            info = {
                "lick": len(self.lick_buffer),
                "cue": self.cue_on,
                "done": False,
                "outcome": self.outcome_type,
            }
        
        # Update the delivered_reward for the observation
        self.delivered_reward = delivered_reward
        self.time += 1

        # === After trial finishes ===
        if info.get("done", False) and info.get("outcome") == "trial_end":
            self.trial_count += 1
            level_up = False
            if self.continual_learning and self.trial_count % self.change_interval == 0 and self.trial_count >= self.change_start:
                self.level += 1
                level_up = True
            info["level_up"] = level_up
        else:
            info["level_up"] = False
    
        return self._get_obs(), delivered_reward, terminated, truncated, info