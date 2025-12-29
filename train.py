import os
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from env.cloud_env import CloudResourceEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    env = CloudResourceEnv(num_servers=5)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path="./logs/checkpoints/",
        name_prefix="cloud_dqn"
    )

    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-3,
        tensorboard_log="./logs/tensorboard/"
    )

    logger.info("Starting training for 100,000 steps...")
    model.learn(
        total_timesteps=100000, 
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save("final_cloud_dqn_model")
    logger.info("Training complete! Model saved as 'final_cloud_dqn_model.zip'")

if __name__ == "__main__":
    train()