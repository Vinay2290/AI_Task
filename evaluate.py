import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from env.cloud_env import CloudResourceEnv

def run_evaluation(model_path, num_steps=1000):
    env = CloudResourceEnv(num_servers=5)
    model = DQN.load(model_path)
    
    obs, _ = env.reset()
    
    # Storage for metrics
    rewards = []
    utilizations = []
    latencies = []
    success_count = 0

    print(f"Running evaluation for {num_steps} steps...")

    for _ in range(num_steps):
        # AI chooses the best server
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        utilizations.append(info['utilization'])
        latencies.append(info['latency'])
        
        if info['latency'] < 1.0: 
            success_count += 1

    avg_util = np.mean(utilizations) * 100
    avg_lat = np.mean(latencies)
    success_rate = (success_count / num_steps) * 100

    print("\n--- EVALUATION RESULTS ---")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Server Utilization: {avg_util:.2f}%")
    print(f"Average Task Latency: {avg_lat:.4f}s")
    print("---------------------------\n")

    plt.figure(figsize=(10, 5))
    plt.plot(utilizations[:100], label='Utilization')
    plt.title('Resource Utilization over 100 Steps (AI Agent)')
    plt.xlabel('Step')
    plt.ylabel('Utilization %')
    plt.legend()
    plt.savefig('utilization_metrics.png')
    print("Graph saved as 'utilization_metrics.png'")

if __name__ == "__main__":
    run_evaluation("final_cloud_dqn_model")