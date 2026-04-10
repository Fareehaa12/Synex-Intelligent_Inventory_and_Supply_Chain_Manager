import pandas as pd
import numpy as np
import os

def generate_sku_data(days=365, num_skus=12):
    """
    Generates synthetic demand data for 10+ SKUs with seasonal patterns.
    This fulfills the requirement for multi-SKU scenario analysis.
    """
    np.random.seed(42) # For reproducibility 
    
    # Create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        
    data = {"Day": np.arange(days)}
    
    for i in range(1, num_skus + 1):
        # Base demand + Seasonal Sine Wave + Random Poisson Noise
        # This makes the environment more challenging than simple static rules
        base = np.random.randint(20, 50)
        seasonality = 15 * np.sin(np.linspace(0, 4 * np.pi, days)) 
        noise = np.random.poisson(lam=5, size=days)
        
        # Combine and ensure demand is never negative
        demand = (base + seasonality + noise).clip(min=0).astype(int)
        data[f"SKU_{i}"] = demand
    
    df = pd.DataFrame(data)
    df.to_csv("data/demand_history.csv", index=False)
    print(f"✅ Success: Generated demand data for {num_skus} SKUs in 'data/demand_history.csv'")

if __name__ == "__main__":
    generate_sku_data()