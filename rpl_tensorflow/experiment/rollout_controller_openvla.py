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
        target_object = res_dict["target_object"]
        return initial_obs, desired_goal, initial_achieved_goal, target_object
    
    def step(self, env_id, action):
        payload_dict = {
            "env_id": env_id,
            "action": action.tolist()
        }
        payload_bytes = pickle.dumps(payload_dict)
        res = requests.post(f"{self.base_url}/step", data=payload_bytes)
        res_dict = pickle.loads(res.content)

        obs = res_dict["obs"]
        reward = res_dict["reward"]
        done = res_dict["done"]
        info = res_dict["info"]
        return obs, reward, done, info

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
class RolloutWorker_OpenVLA:

    @store_args
    def __init__(self, episode_id, ddpg_policy, dims, logger, cfg: GenerateConfig, T, rollout_batch_size=1,
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
        assert self.T > 0
        
        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.returns_history = deque(maxlen=history_len) # added by TS

        self.n_episodes = 0
        self.desired_goal = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_ddpg_obs = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_achieved_goal = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.initial_full_obs = [None] * self.rollout_batch_size
        self.target_object = [None] * self.rollout_batch_size
        self.reset_all_rollouts(episode_id)
        self.clear_history()
        self.task_desired_goals = self.get_task_desired_goals_json()
        self.episode_id = episode_id
        
    def get_task_desired_goals_json(self):
        with open("/home/miki/residual-policy-learning/data/spatial_task_desired_goals.json", "r") as f:
            data = json.load(f)
            return data
    
    # i番目の、環境変数を初期値に戻す
    def reset_rollout(self, task_id, episode_id):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        
        # ここを環境の初期値
        initial_obs, desired_goal, initial_achieved_goal, target_object = self.api.reset(task_id, episode_id)
        self.initial_ddpg_obs[task_id] = self.get_ddpg_obs(initial_obs, target_object)
        self.initial_full_obs[task_id] = initial_obs
        self.desired_goal[task_id] = desired_goal
        self.initial_achieved_goal[task_id] = initial_achieved_goal
        self.target_object[task_id] = target_object
        
    def reset_all_rollouts(self, episode_id):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for task_id in range(10):
            self.reset_rollout(task_id, episode_id)

    def generate_rollouts(self, episode_id):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        rollout_batch_size分のエピソードを作成
        """
        self.episode_id = episode_id
        print("reset")
        self.reset_all_rollouts(self.episode_id)
        
        o = [None] * 10  # observations
        ag = [None] * 10
        
        o[:] = [self.get_ddpg_obs(obs, target_object) for obs, target_object in zip(self.initial_full_obs, self.target_object)]

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_ddpg_obs
        ag[:] = self.initial_achieved_goal

        # generate episodes
        # 1エピソードの各タイムステップ
        obs, achieved_goals, acts, goals, successes, rewards = [], [], [], [], [], []
        Qs = []
        
        for t in range(5):
            print("step", t)
            # actorから得られたΔaction
            residual_action = self.ddpg_policy.get_delta_actions_and_Q(
                    o, ag, self.desired_goal,
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
            for task_id in range(self.rollout_batch_size):
                # episode_id = 0
                base_action = self.api.get_base_action(task_id, 0, self.initial_full_obs[task_id])
                base_u.append(base_action)

            # --- 合成アクション ---
            final_u = base_u + delta_u
            final_u = np.clip(final_u, -self.ddpg_policy.max_u, self.ddpg_policy.max_u)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            reward = np.zeros(self.rollout_batch_size) # added by TS #finger を使え加えるか?
            # # compute new states and observations
            for task_id in range(self.rollout_batch_size):
                target_object = self.target_object[task_id]
                target_object_pos = target_object.replace("_main", "_pos")
                target_object_quat = target_object.replace("_main", "_quat")
                current_obs, _, done, info = self.api.step(task_id, final_u[task_id])
                achieved_goal = np.concatenate([current_obs[target_object_pos], current_obs[target_object_quat]])
                reward[task_id] = -np.linalg.norm(achieved_goal - self.desired_goal[task_id])
                o_new[task_id] = self.get_ddpg_obs(current_obs, target_object)
                ag_new[task_id] = achieved_goal
                
                # success
                if done and success[task_id] == 0.0:
                    success[task_id] = 1.0
                    
            
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            rewards.append(reward.copy()) # added by TS
            acts.append(final_u.copy())
            goals.append(self.desired_goal.copy())
            successes.append(success.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_ddpg_obs[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        
        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        print("success_rate")
        print(success_rate)
        self.success_history.append(success_rate)
        self.returns_history.append(np.mean(np.sum(rewards, axis=0))) # added by TS
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size
        
        return convert_episode_to_batch_major(episode)


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
        actor_weights = self.ddpg_policy.get_actor_weights()
        with open(path, 'wb') as f:
            pickle.dump(actor_weights, f)

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
    
    def get_ddpg_obs(self, obs, target_object):
        obs_list = []

        # ロボット情報
        obs_list.append(obs['robot0_joint_pos'])
        obs_list.append(obs['robot0_joint_vel'])
        obs_list.append(obs['robot0_eef_pos'])
        obs_list.append(obs['robot0_eef_quat'])
        obs_list.append(obs['robot0_gripper_qpos'])
        obs_list.append(obs['robot0_gripper_qvel'])

        obs_list.append(obs[target_object.replace("_main", "_pos")])
        obs_list.append(obs[target_object.replace("_main", "_quat")])

        # flattenして1次元ベクトルに
        ddpg_obs = np.concatenate([x.flatten() for x in obs_list])
        
        return ddpg_obs
