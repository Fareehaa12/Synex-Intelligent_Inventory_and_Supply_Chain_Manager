from src.env.inventory_env import MultiSKUInventoryEnv
import numpy as np

# 1. Initialize the environment
env = MultiSKUInventoryEnv("data/demand_history.csv")
obs, info = env.reset()

print("--- 🏁 STARTING WAREHOUSE STRESS TEST: ORDER NOTHING ---")
print(f"Initial Stock for SKU_1: {obs[0]}")
print(f"Initial Rolling Demand for SKU_1: {obs[12]}")

# 2. Simulate an Action: Order 0 units for every SKU
# Action index '0' maps to 0 units in our self.action_mapping
test_action = np.array([0] * 12) 

# 3. Step through 1 day
new_obs, reward, done, truncated, info = env.step(test_action)

# Extract specific feature indices for SKU_1
# Features: [Stock(0), Demand(12), Pending(24), DaysRemaining(36), Cost(48)]
new_stock_sku_1 = new_obs[0]
pending_sku_1 = new_obs[24]

print("\n--- 📊 RESULTS AFTER 1 DAY ---")
print(f"Action Taken: Ordered 0 units (Index 0) for all SKUs")
print(f"Reward (Profit/Loss): ${reward:,.2f}")
print(f"New Stock for SKU_1: {new_stock_sku_1}")
print(f"Pending Orders for SKU_1: {pending_sku_1}")

# 4. ⚖️ Evaluation & Verdict
print("\n--- ⚖️ VERDICT ---")

# Check 1: Reward should be positive (Profit from sales, no spending)
if reward > 0:
    print(f"✅ SUCCESS: Reward is positive (${reward:,.2f}). We made profit without spending!")
else:
    print(f"❌ WARNING: Reward is negative. Check if holding costs are too high.")

# Check 2: Pending Orders should be zero
if pending_sku_1 == 0:
    print("✅ SUCCESS: Pending orders are 0.0 as expected.")
else:
    print(f"❌ ERROR: Expected 0 pending units, but got {pending_sku_1}.")

# Check 3: Stock should have decreased
if new_stock_sku_1 < obs[0]:
    print(f"✅ SUCCESS: Stock decreased from {obs[0]} to {new_stock_sku_1} due to sales.")
else:
    print("❌ ERROR: Stock did not decrease. Check if demand is being processed.")

# 5. 🔮 LSTM AUXILIARY CHECK
forecast_tomorrow = new_obs[60:72]
print("\n--- 🔮 LSTM AUXILIARY FORECAST ---")
print(f"Predicted Demand for SKU_1 Tomorrow: {forecast_tomorrow[0]:.2f}")
print(f"Full Forecast Vector (Day 1, 12 SKUs): \n{forecast_tomorrow}")

print("\n--- 🧠 ANALYSIS ---")
print("If you ran this for 10 days, your stock would eventually hit 0,")
print("and you would start getting 'Stockout Penalties' (Negative Rewards).")
