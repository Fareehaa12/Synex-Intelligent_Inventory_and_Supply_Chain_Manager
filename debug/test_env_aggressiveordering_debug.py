from src.env.inventory_env import MultiSKUInventoryEnv
import numpy as np

# 1. Initialize the environment
env = MultiSKUInventoryEnv("data/demand_history.csv")
obs, info = env.reset()

print("--- STARTING WAREHOUSE SIMULATION TEST ---")
print(f"Initial Stock for SKU_1: {obs[0]}")
print(f"Initial Rolling Demand for SKU_1: {obs[12]}") # Index 12 is start of demand features

# 2. Simulate an Action: Let's order 100 units of every SKU
# Action index '2' maps to 100 units in our code
test_action = np.array([2] * 12) 

# 3. Step through 1 day
new_obs, reward, done, truncated, info = env.step(test_action)

print("\n--- RESULTS AFTER 1 DAY ---")
print(f"Action Taken: Ordered 100 units for all SKUs")
print(f"Reward (Profit/Loss): ${reward:.2f}")
print(f"New Stock for SKU_1: {new_obs[0]}")
print(f"Pending Orders for SKU_1: {new_obs[24]}") # Index 24 is start of pending features

# 4. Verification Logic
if new_obs[24] == 100:
    print("\n✅ SUCCESS: Lead Time logic is working! (100 units are in 'Pending')")
else:
    print("\n❌ ERROR: Pending orders not showing up correctly.")

if reward != 0:
    print("✅ SUCCESS: Reward calculation is active!")