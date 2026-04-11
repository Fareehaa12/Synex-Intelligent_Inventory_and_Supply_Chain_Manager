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

    # TUNED FOR PERFORMANCE:
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0007,
        n_steps=10,          # Frequent updates to catch demand trends
        gamma=0.95,          # Focuses slightly more on immediate profit
        ent_coef=0.05,       # High exploration to find "Baseline-beating" moves
        verbose=1,
        tensorboard_log="./logs/a2c_tensorboard/"
    )

    print("🔥 TRAINING A2C CHALLENGER for 500,000 steps...")
    
    # Save checkpoints every 50k steps
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./saved_models/a2c/', name_prefix='a2c_tuned_model')
    
    # Increased to 500,000 for mastery
    model.learn(total_timesteps=500000, callback=checkpoint_callback)
    
    model.save("saved_models/a2c_inventory_final")
    print("✅ A2C Training Complete. Ready for the final tournament!")

if __name__ == "__main__":
    train()
