from collections import deque
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import torch

class SessionRecorder:
    """
    Straight‐up session+training recorder with no SB3 callbacks.
    
    Usage:
      recorder = SessionRecorder(lick_action=1)
      
      # in your env loop:
      obs, _ = env.reset()
      while not done:
          action = ...           # sample or predict
          next_obs, reward, done, truncated, info = env.step(action)
          recorder.record_env_step(action, reward, next_obs, info)
          obs = next_obs
      
      # then when you call `model.train(...)`:
      model.train(batch_size=N, gradient_steps=1)
      recorder.record_train(model)
    """
    def __init__(self, lick_action=1):
        self.trial_idx = []
        
        # session‐by‐step logs
        self.td_errors = []
        self.td_pid_errors = []
        self.licks     = []
        self.tones     = []
        self.rewards   = []
        self.losses    = []
        self.enl_breaks = []
        self.dones     = []
        self._prev_obs = None
        self.lick_action = lick_action
        self.omissions = []
        
        # per‐train‐call logs
        self.p  = []
        self.d  = []
        self.i  = []
        self.kp = []
        self.ki = []
        self.kd = []

        # For EPLHb output and coeff
        if not hasattr(self, 'eplhb_out'):
            self.eplhb_out = []
        if not hasattr(self, 'eplhb_coeff'):
            self.eplhb_coeff = []

    def record_env_step(self, trial_idx, action, reward, new_obs, info, model=None):
        """Call right after env.step(...)"""
        import numpy as np
        import torch

        td = 0.0
        td_pid = 0.0
        done = int(info.get("done", False))

        # Convert action to Python int if needed
        if isinstance(action, np.ndarray):
            action = int(action.item())

        if self._prev_obs is not None and model is not None:
            obs_t  = torch.as_tensor(self._prev_obs, dtype=torch.float32).unsqueeze(0).to(model.device)
            next_t = torch.as_tensor(new_obs, dtype=torch.float32).unsqueeze(0).to(model.device)
            a_t    = torch.tensor([[action]], device=model.device, dtype=torch.long)

            with torch.no_grad():
                q_cur  = model.q_net(obs_t)[0, action]           # scalar Q(s, a)
                q_next = model.q_net_target(next_t).max(1)[0]    # max_a' Q'(s', a')

                td_tensor = reward + (1 - done) * model.gamma * q_next - q_cur
                td = td_tensor.item()

                # Get PID gains
                kp, ki, kd, alpha, beta = model.gain_adapter.get_gains(obs_t, a_t, None)

                # Proportional term
                p = td_tensor  # still a tensor

                # Initialize integrator state if missing
                if not hasattr(self, "z_prev"):
                    self.z_prev = torch.tensor(0.0, device=model.device)

                # Integral term
                i = beta * self.z_prev + alpha * p
                self.z_prev = i  # update for next step

                # Derivative term
                d_val = model.d_net(obs_t)  # [1, n_actions]
                d = q_cur - d_val.gather(1, a_t).squeeze(1)  # scalar tensor

                # PID TD error
                td_pid_tensor = kp * p + ki * i + kd * d
                td_pid = td_pid_tensor.item()

                # sanity check: ensure td tensor + PID components = td_pid
                # td_pid_reconstructed = (kp * (td_tensor) + ki * i + kd * (q_cur - d_val.gather(1, a_t).squeeze(1))).item()
                # print("td:", td, "td_pid:", td_pid, "td_pid_reconstructed:", td_pid_reconstructed)

        else:
            if not hasattr(self, "z_prev"):
                self.z_prev = torch.tensor(0.0)

        self._prev_obs = new_obs


        # Flags
        lick_flag     = int(action == self.lick_action)
        tone_flag     = int(info.get("cue", False))
        omission_flag = int(info.get("outcome", False) == "omission")


        # append
        self.td_errors.append(td)
        self.td_pid_errors.append(td_pid)
        self.licks.append(lick_flag)
        self.tones.append(tone_flag)
        self.rewards.append(reward)
        self.dones.append(bool(info.get("done", False)))
        self.omissions.append(omission_flag)

        # 4) record PID gains
        def mean_or_none(x):
            if x is None: return 0.0
            arr = x.detach().cpu().numpy()
            return float(arr.mean())
        
        # grab the raw PID‐DQN fields
        p  = getattr(model, "p_update", None)
        i  = getattr(model, "i_update", None)
        d  = getattr(model, "d_update", None)
        kp = getattr(model, "kp",       None)
        ki = getattr(model, "ki",       None)
        kd = getattr(model, "kd",       None)
        loss = getattr(model, "latest_loss", 0.0)

        # record their batch‐means
        self.p.append(mean_or_none(p))
        self.i.append(mean_or_none(i))
        self.d.append(mean_or_none(d))
        self.kp.append(mean_or_none(kp))
        self.ki.append(mean_or_none(ki))
        self.kd.append(mean_or_none(kd))
        self.losses.append(loss)

        # 5) record the trial index
        self.trial_idx.append(trial_idx)

        # 6) If model is EPLHb_DQN, record EPLHb output and coeff
        if model is not None and hasattr(model, 'policy'):
            q_net = getattr(model.policy, 'q_net', None)
            if q_net is not None and hasattr(q_net, 'forward_full') and hasattr(q_net, 'eplhb_coeff'):
                obs_t = torch.as_tensor(new_obs).unsqueeze(0).to(model.device)
                with torch.no_grad():
                    result = q_net.forward_full(obs_t)
                if result is not None and len(result) == 3:
                    _, _, eplhb_out = result
                    self.eplhb_out.append(float(eplhb_out.item()) if hasattr(eplhb_out, 'item') else float(eplhb_out))
                else:
                    self.eplhb_out.append(None)
                coeff = q_net.eplhb_coeff
                self.eplhb_coeff.append(float(coeff.item()) if hasattr(coeff, 'item') else float(coeff))




class SessionRecorderCallback(BaseCallback):
    """
    Logs at every step:
      - td_error   (computed from q and q_target)
      - lick       (action==lick_action)
      - tone       (info['cue'])
      - reward     (env reward)
      - loss       (most recently logged train/loss)
      - done_flag  (info['done'])
    After training you'll have aligned lists of length = total steps:
      .td_errors, .licks, .tones, .rewards, .losses, .dones
    """
    def __init__(self, lick_action=1, verbose=0):
        super().__init__(verbose)
        # Decouples the logger from your env's encoding. 
        # If you ever change your env so that "lick" is action 2 instead of 1, 
        # you just pass lick_action=2 into the callback—no need to rewrite any logging code.
        self.lick_action = lick_action

        # time‐series logs
        # (these are all lists of length = total steps)
        self.td_errors = []
        self.licks     = []
        self.tones     = []
        self.rewards   = []
        self.losses    = []
        self.dones     = []

        # pid logs
        self.p  = []
        self.d  = []
        self.i  = []
        self.kp = []
        self.ki = []
        self.kd = []

        self._prev_obs = None

    # helper to pull a scalar mean (or None)
    def mean_or_none(self, x):
        if x is None:
            return 0
        # x might be a torch.Tensor of shape (batch_size,1) or similar
        arr = x.detach().cpu().numpy()
        return float(arr.mean())

    def _on_step(self) -> bool:
        # pull out the locals
        actions =  self.locals.get("actions")
        rewards =  self.locals.get("rewards")
        new_obs  =  self.locals.get("new_obs")
        infos    =  self.locals.get("infos")

        # pull out update term
        p_update = getattr(self.model, "p_update", None)
        d_update = getattr(self.model, "d_update", None)
        i_update = getattr(self.model, "i_update", None)
        kp      = getattr(self.model, "kp", None)
        ki      = getattr(self.model, "ki", None)
        kd      = getattr(self.model, "kd", None)

        # take the mean
        p_update = self.mean_or_none(p_update)
        d_update = self.mean_or_none(d_update)
        i_update = self.mean_or_none(i_update)
        kp      = self.mean_or_none(kp)
        ki      = self.mean_or_none(ki)
        kd      = self.mean_or_none(kd)

        # unpack single-env
        action = actions[0] if isinstance(actions, (list, np.ndarray)) else actions
        reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
        info   = infos[0]   if isinstance(infos, list)          else infos

        # 1) compute TD error
        td = 0.0
        done = int(info.get("done", False))
        if self._prev_obs is not None:
            # turn to torch tensors
            obs_t     = torch.as_tensor(self._prev_obs).unsqueeze(0).to(self.model.device)
            next_obs_t= torch.as_tensor(new_obs).unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                # Q(cur, a)
                q_cur  = self.model.q_net(obs_t)[0, action]
                # max_a' Q_target(next, a')
                q_next = self.model.q_net_target(next_obs_t).max(dim=1)[0]
                td = (reward + (1-done) * self.model.gamma * q_next - q_cur).item()
        self._prev_obs = new_obs

        # 2) binary lick flag
        lick = 1 if int(action) == self.lick_action else 0

        # 3) binary tone flag (assumes env.info['cue']==True on tone)
        tone = 1 if info.get("cue", False) else 0

        # 4) loss
        loss = self.model.logger.name_to_value.get("train/loss", 0.0)

        # 4) append to your session logs
        self.td_errors.append(td)
        self.licks    .append(lick)
        self.tones    .append(tone)
        self.rewards  .append(reward)
        self.losses   .append(loss)
        self.dones    .append(bool(info.get("done", False)))

        # 5) append to your pid logs
        self.p.append(p_update)
        self.d.append(d_update)
        self.i.append(i_update)
        self.kp.append(kp)
        self.ki.append(ki)
        self.kd.append(kd)

        return True


class TrialRecorderCallback(BaseCallback):
    """
    Records, for each trial:
     - the last `pre_steps` actions before the cue,
     - the next `post_steps` TD-errors after the cue,
     pads to exactly pre+post length,
     and collects them in `self.all_licks` and `self.all_tds`.
    Relies on your env.info having:
      • info['cue']==True exactly on the cue step,
      • info['done']==True exactly at trial end.
    """
    def __init__(self, lick_action=1, pre_steps=20, post_steps=30, verbose=0):
        super().__init__(verbose)
        self.lick_action    = lick_action
        self.pre_steps      = pre_steps
        self.post_steps     = post_steps
        self.max_len        = pre_steps + post_steps

        # circular buffers to hold the last `pre_steps` at any time
        self.pre_actions = deque([0]*pre_steps, maxlen=pre_steps)
        self.pre_tds     = deque([0.0]*pre_steps, maxlen=pre_steps)

        # flags & working lists for the current trial
        self.recording    = False
        self.trial_actions = []
        self.trial_tds     = []
        self.prev_obs      = None

        # results across trials
        self.all_licks      = []
        self.all_tds        = []
        self.reward_history = []   # you can also record these if you like
        self.loss_history   = []

    def _on_step(self) -> bool:
        # 1) pull out SB3 locals
        actions = self.locals.get("actions")
        new_obs = self.locals.get("new_obs")
        rewards = self.locals.get("rewards")
        infos   = self.locals.get("infos")
        

        # single‐env unpack
        action = actions[0] if isinstance(actions, (list, np.ndarray)) else actions
        reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
        info   = infos[0]   if isinstance(infos, list)          else infos

        # 2) compute the instantaneous TD‐error exactly like your original callback
        td_err = 0.0
        done = int(info.get('done', False))
        if self.prev_obs is not None:
            obs_t  = torch.as_tensor(self.prev_obs).unsqueeze(0).to(self.model.device)
            next_t = torch.as_tensor(new_obs).unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                q_cur  = self.model.q_net(obs_t)[0, action]
                q_next = self.model.q_net_target(next_t).max(1)[0]
                td_err = (reward + (done-1) * self.model.gamma * q_next - q_cur).item()
        self.prev_obs = new_obs

        # 3) update the pre‐cue buffers if not yet recording
        if not self.recording:
            self.pre_actions.append(int(action == self.lick_action))
            self.pre_tds.append(td_err)

        # 4) detect cue onset
        if not self.recording and info.get("cue", False):
            self.recording = True
            # seed the trial lists from the pre‐cue buffer
            self.trial_actions = list(self.pre_actions)
            self.trial_tds     = list(self.pre_tds)

        # 5) while recording, keep collecting
        elif self.recording:
            self.trial_actions.append(int(action == self.lick_action))
            self.trial_tds.append(td_err)

        # 6) on trial end, pad & store
        if self.recording and info.get("done", False):
            L = len(self.trial_actions)
            if L < self.max_len:
                self.trial_actions += [0]*(self.max_len - L)
                self.trial_tds     += [0.0]*(self.max_len - L)

            # save fixed‐length arrays
            self.all_licks.append(np.array(self.trial_actions[:self.max_len]))
            self.all_tds.append(np.array(self.trial_tds[:self.max_len]))

            # (optional) record reward & loss
            self.reward_history.append(reward)
            loss = self.model.logger.name_to_value.get("train/loss")
            self.loss_history.append(loss if loss is not None else 0.0)

            # reset for next trial
            self.recording    = False
            self.pre_actions = deque([0]*self.pre_steps, maxlen=self.pre_steps)
            self.pre_tds     = deque([0.0]*self.pre_steps, maxlen=self.pre_steps)

        return True
    
class TrialLimitCallback(BaseCallback):
    def __init__(self, max_trials, verbose=1):
        super().__init__(verbose)
        self.max_trials = max_trials
        self.trial_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            done = info.get("done")
            if done:
                # an episode just finished
                self.trial_count += 1
                print(f"\nTrial {self.trial_count}/{self.max_trials} finished")
                if self.trial_count >= self.max_trials:
                    print("--- Reached max trials, stopping training ---")
                    return False  # returning False tells .learn() to stop
        return True
