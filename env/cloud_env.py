import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CloudResourceEnv(gym.Env):
    def __init__(self, num_servers=5):
        super(CloudResourceEnv, self).__init__()
        self.num_servers = num_servers
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_servers * 2 + 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_servers)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.server_loads = np.zeros((self.num_servers, 2), dtype=np.float32) 
        self.current_task = self._generate_task()
        return self._get_obs(), {}

    def _generate_task(self):
        return np.random.uniform(0.05, 0.3, size=(2,)).astype(np.float32)

    def _get_obs(self):
        return np.concatenate([self.server_loads.flatten(), self.current_task])

    def step(self, action):
        target_server = action
        task_req = self.current_task
        
        if np.all(self.server_loads[target_server] + task_req <= 1.0):
            self.server_loads[target_server] += task_req
            reward = 1.0 
            latency = 0.1
        else:
            reward = -1.0
            latency = 1.0

        avg_util = np.mean(self.server_loads)
        reward += avg_util

        reward = float(reward) 

        self.server_loads = np.clip(self.server_loads - 0.03, 0, 1)
        
        self.current_task = self._generate_task()
        
        info = {"utilization": float(avg_util), "latency": float(latency)}
        
        return self._get_obs(), reward, False, False, info