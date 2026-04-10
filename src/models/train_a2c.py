import os
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from src.env.inventory_env import MultiSKUInventoryEnv

def train():
    env = MultiSKUInventoryEnv("data/demand_history.csv")
    env = Monitor(env) 
    
    os.makedirs("saved_models/a2c/", exist_ok=True)
    os.makedirs("logs/a2c_tensorboard/", exist_ok=True)

    # A2C supports MultiDiscrete action spaces!
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/a2c_tensorboard/"
    )

    print("🚀 Training A2C Agent for 100,000 steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path='./saved_models/a2c/', 
        name_prefix='a2c_inv_model'
    )
    
    model.learn(total_timesteps=100000, callback=checkpoint_callback)
    model.save("saved_models/a2c_inventory_final")
    print("✅ A2C Training Complete. Model saved in saved_models/")

if __name__ == "__main__":
    train()