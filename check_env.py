from stable_baselines3.common.env_checker import check_env
from env.cloud_env import CloudResourceEnv

def verify():
    env = CloudResourceEnv()
    # It will throw an error if the environment has issues
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")

if __name__ == "__main__":
    verify()