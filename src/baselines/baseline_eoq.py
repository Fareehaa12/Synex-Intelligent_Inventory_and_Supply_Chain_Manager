import pandas as pd
import numpy as np
import json
import os
from src.env.inventory_env import MultiSKUInventoryEnv

# --- PHASE 1 & 2: THE MATH & LOGIC ---
def calculate_eoq_params(df, sku_index):
    """
    Calculates the Economic Order Quantity (EOQ) and Reorder Point (ROP).
    Phase 1: EOQ formula balances holding vs ordering costs.
    Phase 2: ROP ensures we order before stock hits zero.
    """
    # Industry Constants
    ORDER_COST = 25.0   # (S) Cost per order
    HOLDING_COST = 0.5  # (H) Cost to hold 1 unit for 1 day
    LEAD_TIME = 3       # Days it takes for an order to arrive
    
    col_name = f"SKU_{sku_index + 1}"
    avg_demand = df[col_name].mean()
    std_demand = df[col_name].std()
    
    # EOQ = sqrt((2 * Annual Demand * Order Cost) / Annual Holding Cost)
    # We use 365 days for annual scaling
    annual_demand = avg_demand * 365
    annual_holding_cost = HOLDING_COST * 365
    
    eoq = np.sqrt((2 * annual_demand * ORDER_COST) / annual_holding_cost)
    
    # ROP = (Daily Demand * Lead Time) + Safety Stock
    # Safety Stock for 95% service level (1.645)
    safety_stock = 1.645 * std_demand * np.sqrt(LEAD_TIME)
    rop = (avg_demand * LEAD_TIME) + safety_stock
    
    return round(eoq), round(rop)

# --- PHASE 3: THE SIMULATION ---
def run_baseline_simulation():
    """
    Runs a 365-day simulation using the EOQ logic to establish a benchmark.
    """
    env = MultiSKUInventoryEnv("data/demand_history.csv")
    df = pd.read_csv("data/demand_history.csv")
    obs, _ = env.reset()
    
    # Pre-calculate EOQ/ROP for all 12 SKUs
    sku_params = []
    for i in range(12):
        q, rop = calculate_eoq_params(df, i)
        sku_params.append({'q': q, 'rop': rop})
    
    total_reward = 0
    done = False
    day_count = 0
    
    print(f"--- 🚀 STARTING EOQ BASELINE SIMULATION ---")
    
    while not done:
        actions = []
        for i in range(12):
            current_stock = obs[i]
            # EOQ Decision Rule: If stock <= Reorder Point, buy the EOQ amount
            if current_stock <= sku_params[i]['rop']:
                # Map units to the closest environment action index (step of 50)
                action_idx = min(round(sku_params[i]['q'] / 50), 10)
                actions.append(action_idx)
            else:
                actions.append(0) # Do nothing
        
        obs, reward, done, truncated, _ = env.step(np.array(actions))
        total_reward += reward
        day_count += 1
        
        if day_count % 100 == 0:
            print(f"Day {day_count}/365 | Current Profit: ${total_reward:,.2f}")

    # --- PHASE 4: ANALYTICS (SAVING RESULTS) ---
    save_results(total_reward, sku_params)
    
    return total_reward

def save_results(final_profit, params):
    """Saves the benchmark data to a JSON file for later comparison."""
    os.makedirs('results', exist_ok=True)
    
    output = {
        "benchmark_name": "EOQ_Traditional_Math",
        "total_yearly_profit": round(final_profit, 2),
        "avg_daily_profit": round(final_profit / 365, 2),
        "sku_configurations": params
    }
    
    with open('results/eoq_results.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print("\n--- 🏁 BASELINE COMPLETE ---")
    print(f"Total Yearly Profit: ${final_profit:,.2f}")
    print(f"Results exported to 'results/eoq_results.json'")

if __name__ == "__main__":
    run_baseline_simulation()