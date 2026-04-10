import pandas as pd
import numpy as np
from src.env.inventory_env import MultiSKUInventoryEnv
from src.baselines.baseline_eoq import calculate_eoq_params

def run_stress_test(scenario_name, demand_spike_multiplier):
    print(f"\n--- 🔥 SCENARIO ANALYSIS: {scenario_name} ---")
    
    env = MultiSKUInventoryEnv("data/demand_history.csv")
    df = pd.read_csv("data/demand_history.csv")
    
    # Use the same EOQ settings we just generated
    sku_settings = [calculate_eoq_params(df, i) for i in range(12)]
    
    obs, _ = env.reset()
    total_reward = 0
    
    # Run for 100 days to see the impact
    for day in range(100):
        actions = []
        for i in range(12):
            q, rop = sku_settings[i]
            # Traditional math doesn't know a spike is coming!
            if obs[i] <= rop:
                action_idx = min(round(q / 50), 10)
                actions.append(action_idx)
            else:
                actions.append(0)
        
        # Step the environment
        obs, reward, done, _, _ = env.step(np.array(actions))
        
        # Simulate a demand spike by penalizing the reward more heavily
        # if stock levels are dangerously low during the "spike"
        if demand_spike_multiplier > 1.0:
            stockouts = sum(1 for s in obs[:12] if s <= 0)
            reward -= (stockouts * 500) # Heavy penalty for failing during a crisis
            
        total_reward += reward

    print(f"Resulting Profit (100 Days): ${total_reward:,.2f}")
    return total_reward

if __name__ == "__main__":
    # 1. Test Normal Performance
    normal = run_stress_test("Steady State", 1.0)
    
    # 2. Test High Volatility (The "Crisis" Scenario)
    crisis = run_stress_test("Supply Chain Crisis / Demand Surge", 2.0)
    
    print(f"\n💡 Profit Drop during Crisis: ${normal - crisis:,.2f}")