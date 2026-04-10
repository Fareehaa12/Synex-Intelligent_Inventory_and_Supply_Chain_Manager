import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from src.env.inventory_env import MultiSKUInventoryEnv

def train():
    # 1. Setup Environment
    env = MultiSKUInventoryEnv("data/demand_history.csv")
    env = Monitor(env) # Tracks metrics for logging
    
    # 2. Create Folders for Output
    os.makedirs("saved_models/ppo/", exist_ok=True)
    os.makedirs("logs/ppo_tensorboard/", exist_ok=True)

    # 3. Define the PPO Agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99, # Discount factor
        verbose=1,
        tensorboard_log="./logs/ppo_tensorboard/"
    )

    print("🚀 Training PPO Agent for 100,000 steps...")
    
    # 4. Train with a checkpoint saver (saves every 20k steps)
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='./saved_models/ppo/', name_prefix='ppo_inv_model')
    
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # 5. Save the Final Model
    model.save("saved_models/ppo_inventory_final")
    print("✅ PPO Training Complete. Model saved in saved_models/")

if __name__ == "__main__":
    train()