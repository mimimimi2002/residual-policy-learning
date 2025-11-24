from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from baselines.her.util import convert_episode_to_batch_major, store_args
import pdb
import json

# for GenerateConfig
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    libero_raw_data_dir: str = "/home/miki/LIBERO/libero_dataset/datasets/libero_spatial"

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    

    # fmt: on

import requests
import json
import numpy as np

class EnvAPIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def reset(self, task_id, episode_id):
        res = requests.post(f"{self.base_url}/reset", json={"task_id": task_id, "episode_id": episode_id})
        res_dict = pickle.loads(res.content)

        initial_obs = res_dict["initial_obs"]
        desired_goal = res_dict["desired_goal"]
        initial_achieved_goal = res_dict["initial_achieved_goal"]
        return initial_obs, desired_goal, initial_achieved_goal
    
    def step(self, env_id, action):
        res = requests.post(f"{self.base_url}/step", json={
            "env_id": env_id,
            "action": action.tolist()
        })
        return res.json()

    def set_init_state(self, env_id, init_state):
        res = requests.post(f"{self.base_url}/set_init_state", json={
            "env_id": env_id,
            "init_state": init_state
        })
        return res.json()

    def get_base_action(self, task_id, episode_id, obs):
        payload_dict = {
            "task_id": task_id,
            "episode_id": episode_id,
            "obs": obs,  # OrderedDictそのまま
        }
        payload_bytes = pickle.dumps(payload_dict)
        res = requests.post(f"{self.base_url}/get_base_action", data=payload_bytes)
        action_dict = pickle.loads(res.content)
        action = np.array(action_dict["action"])
        return action


#あるtask_idであるepisode_idの時
class RolloutWorker:

    @store_args
    def __init__(self, make_env, ddpg_policy, dims, logger, cfg: GenerateConfig, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, controller_prop=0,history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            ddpg_policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.api = EnvAPIClient("http://localhost:8000")
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.returns_history = deque(maxlen=history_len) # added by TS

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.initial_obs = [None] * 10 #置き換え
        self.initial_achieved_goal = [None] * 10
        self.initial_desired_goal = [None] * 10
        self.reset_all_rollouts()
        self.clear_history()
        self.task_desired_goals = self.get_task_desired_goals_json()
        
    def get_task_desired_goals_json(self):
        with open("/home/miki/residual-policy-learning/data/spatial_task_desired_goals.json", "r") as f:
            data = json.load(f)
            return data

    # i番目の、環境変数を初期値に戻す
    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        
        # ここを環境の初期値
        # desired_goalはここでしか撮らない
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']
    
    # i番目の、環境変数を初期値に戻す
    def reset_rollout2(self, task_id, episode_id):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        
        # ここを環境の初期値
        initial_obs, desired_goal, initial_achieved_goal = self.api.reset(task_id, episode_id)
        self.initial_obs[task_id] = initial_obs
        self.initial_desired_goal[task_id] = desired_goal
        self.initial_achieved_goal[task_id] = initial_achieved_goal
        

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)
    
    def reset_all_rollouts2(self, episode_id):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for task_id in range(10):
            self.reset_rollout2(task_id, episode_id)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        rollout_batch_size分のエピソードを作成
        """
        
        # 環境の初期化で、initial_obs, initial_achived_goal, inital_desired_goalが初期化
        # 通信必要
        self.reset_all_rollouts()
        
        # get base action
        # base_action = self.api.get_base_action(0, 0, initial_obs)

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        # 1エピソードの各タイムステップ
        obs, achieved_goals, acts, goals, successes, rewards = [], [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        
        # タイムステップ
        for t in range(self.T):
            # actorから得られたΔaction
            residual_action = self.ddpg_policy.get_delta_actions_and_Q(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                controller_prop=self.controller_prop if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                delta_u, Q = residual_action
                Qs.append(Q)
            else:
                delta_u = residual_action

            if delta_u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                delta_u = delta_u.reshape(1, -1)

            # 実際に環境とinteractionして
            # 既存のopenvlaでbaseのactionを生成
            # obsを更新
            # 通信必要
            base_u = []
            # for i in range(self.rollout_batch_size):
            #     with torch.no_grad():
            #         obs_tensor = self._preprocess_for_openvla(o[i])  # 必要なら画像→テンソル変換
            #         base_action = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) #self.openvla_policy.predict_action(obs_tensor)
            #         base_u.append(base_action)

            # --- 合成アクション ---
            # final_u = base_u + delta_u
            final_u = delta_u
            final_u = np.clip(final_u, -self.ddpg_policy.max_u, self.ddpg_policy.max_u)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            reward = np.zeros(self.rollout_batch_size) # added by TS
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    # obs, reward, done, info = env.step(action.tolist())
                    #　通信必要
                    curr_o_new, r, _, info = self.envs[i].step(final_u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    reward[i] = r # Added by TS
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            rewards.append(reward.copy()) # added by TS
            acts.append(final_u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        self.returns_history.append(np.mean(np.sum(rewards, axis=0))) # added by TS
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)
    
    def generate_rollouts2(self, episode_id):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        rollout_batch_size分のエピソードを作成
        """
        
        self.reset_all_rollouts2(episode_id)
        
    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()
        self.returns_history.clear() # added by TS

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.ddpg_policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('returns', np.mean(self.returns_history))] # added by TS
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
