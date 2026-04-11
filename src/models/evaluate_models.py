import json
import os
import pandas as pd
from stable_baselines3 import PPO, A2C
from src.env.inventory_env import MultiSKUInventoryEnv

def run_eval(model_path, model_class):
    """Runs a 1-year simulation and returns the total profit."""
    env = MultiSKUInventoryEnv("data/demand_history.csv")
    
    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        return None
        
    model = model_class.load(model_path)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if truncated:
            break
            
    return total_reward

if __name__ == "__main__":
    print("\n" + "="*40)
    print("🧪 STARTING HEAD-TO-HEAD EVALUATION")
    print("="*40)

    # 1. Load EOQ Baseline Result
    try:
        with open("results/eoq_results.json", "r") as f:
            eoq_data = json.load(f)
            eoq_profit = eoq_data["total_yearly_profit"]
    except FileNotFoundError:
        eoq_profit = 0
        print("⚠️ EOQ results not found. Please run the baseline script first.")

    # 2. Run AI Evaluations
    print("🤖 Evaluating PPO Agent...")
    ppo_profit = run_eval("saved_models/ppo_inventory_final", PPO)
    
    print("🤖 Evaluating A2C Agent...")
    a2c_profit = run_eval("saved_models/a2c_inventory_final", A2C)

    # 3. Display Results
    print("\n" + "📊 FINAL PERFORMANCE COMPARISON")
    print("-" * 40)
    print(f"{'METHOD':<20} | {'TOTAL YEARLY PROFIT':<20}")
    print("-" * 40)
    print(f"{'Mathematical (EOQ)':<20} | ${eoq_profit:,.2f}")
    
    if ppo_profit is not None:
        print(f"{'RL Agent (PPO)':<20} | ${ppo_profit:,.2f}")
    
    if a2c_profit is not None:
        print(f"{'RL Agent (A2C)':<20} | ${a2c_profit:,.2f}")
    print("-" * 40)

    # 4. Final Verdict
    results = {"EOQ": eoq_profit}
    if ppo_profit: results["PPO"] = ppo_profit
    if a2c_profit: results["A2C"] = a2c_profit
    
    winner = max(results, key=results.get)
    improvement = ((results[winner] - eoq_profit) / abs(eoq_profit)) * 100 if eoq_profit != 0 else 0
    
    print(f"\n🏆 WINNER: {winner}")
    if winner != "EOQ":
        print(f"📈 Improvement over Baseline: {improvement:.2f}%")
    print("="*40 + "\n")