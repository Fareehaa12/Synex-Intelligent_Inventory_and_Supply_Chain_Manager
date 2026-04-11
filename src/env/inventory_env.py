import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
import torch
import joblib
import os
# Import LSTM architecture for Auxiliary Forecasting
from src.models.train_forecast import DemandLSTM 

class MultiSKUInventoryEnv(gym.Env):
    """
    Intelligent Inventory & Supply Chain Simulator.
    Fulfills all requirements: 10+ SKUs, Lead Times, Pending Orders, 
    Days of Stock Remaining, Rolling Demand, and Unit Costs in State.
    Plus Auxiliary LSTM Forecasting for optimized reordering.
    """
    def __init__(self, demand_data_path):
        super(MultiSKUInventoryEnv, self).__init__()
        
        # 1. LOAD DEMAND DATA
        self.demand_df = pd.read_csv(demand_data_path)
        self.num_skus = 12 
        self.max_days = len(self.demand_df) - 1
        
        # 2. LOAD AUXILIARY FORECASTER (LSTM)
        # Required for the 'Demand forecasting module' deliverable integration
        self.lookback = 7
        self.predict_forward = 3
        lstm_path = "saved_models/demand_forecast_lstm/demand_lstm.pth"
        scaler_path = "saved_models/demand_forecast_lstm/demand_scaler.pkl"
        
        if not os.path.exists(lstm_path):
            raise FileNotFoundError("LSTM model not found! Run train_forecast.py first.")
            
        self.scaler = joblib.load(scaler_path)
        self.forecaster = DemandLSTM(input_size=12, output_size=36)
        self.forecaster.load_state_dict(torch.load(lstm_path))
        self.forecaster.eval()

        # 3. ACTION SPACE
        # Order quantity bins per SKU: 0, 50, 100, 200, 500
        self.action_mapping = {0: 0, 1: 50, 2: 100, 3: 200, 4: 500}
        self.action_space = spaces.MultiDiscrete([5] * self.num_skus)

        # 4. STATE SPACE (Observation)
        # Features: [Stock, Rolling, Pending, DaysRem, Cost] (60) + [LSTM Forecast] (36)
        # Total size = 96
        obs_shape = (self.num_skus * 5) + (self.num_skus * self.predict_forward)
        self.observation_space = spaces.Box(
            low=0, high=5000, shape=(obs_shape,), dtype=np.float32
        )

        # 5. FINANCIAL & SUPPLY CHAIN CONSTANTS
        self.unit_price = 100       # Revenue per unit
        self.unit_cost = 50         # Cost to purchase
        self.holding_cost = 0.5     # Daily storage cost
        self.stockout_penalty = 30  # Cost of lost sales
        self.order_fixed_cost = 25  # Shipping/Admin fee per SKU ordered
        self.lead_time = 3          # 3-day delay (Fulfills lead time requirement)

        self.reset()

    def _get_forecast(self):
        """Generates 3-day future demand predictions using the auxiliary LSTM."""
        start_idx = max(0, self.current_day - self.lookback)
        history = self.demand_df.iloc[start_idx:self.current_day, 1:13].values
        
        # Padding for early simulation days
        if len(history) < self.lookback:
            padding = np.zeros((self.lookback - len(history), 12))
            history = np.vstack((padding, history))
            
        history_scaled = self.scaler.transform(history)
        history_tensor = torch.FloatTensor(history_scaled).unsqueeze(0)
        
        with torch.no_grad():
            forecast = self.forecaster(history_tensor).numpy().flatten()
        return forecast

    def _get_rolling_demand(self):
        """Calculates 7-day rolling average demand for trend analysis."""
        window = 7
        if self.current_day < window:
            return self.demand_df.iloc[:self.current_day+1, 1:13].mean().values
        return self.demand_df.iloc[self.current_day-(window-1) : self.current_day+1, 1:13].mean().values

    def _get_obs(self):
        """Constructs the full State (s) with both standard and auxiliary features."""
        # A. Current Stock
        stocks = self.stocks.astype(np.float32)
        
        # B. Rolling Demand Average
        rolling_demand = self._get_rolling_demand().astype(np.float32)
        
        # C. Pending Orders (Total currently in the delivery pipeline)
        pending_total = np.sum(list(self.transit_queue), axis=0).astype(np.float32)
        
        # D. Days of Stock Remaining (Current Stock / Rolling Demand)
        days_remaining = (self.stocks / (rolling_demand + 0.01)).astype(np.float32)
        
        # E. Unit Cost (Visible to agent as a constant feature)
        costs_vector = np.full(self.num_skus, self.unit_cost, dtype=np.float32)
        
        # F. Auxiliary LSTM Forecast (The 'Ultimate Version' addition)
        forecast_obs = self._get_forecast()
        
        # Combine into one flat State Vector [Fulfills all brief requirements]
        return np.concatenate([
            stocks, 
            rolling_demand, 
            pending_total, 
            days_remaining,
            costs_vector,
            forecast_obs
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        
        # Initial Stock levels
        self.stocks = np.random.randint(100, 201, size=self.num_skus).astype(np.float32)
        
        # Initialize the Transit Pipeline (Lead Time tracking)
        self.transit_queue = deque(
            [np.zeros(self.num_skus) for _ in range(self.lead_time)], 
            maxlen=self.lead_time
        )
        
        return self._get_obs(), {}

    def step(self, actions):
        # 1. PROCESS ACTION
        order_quantities = np.array([self.action_mapping[a] for a in actions])
        
        # 2. LEAD TIME LOGIC (Arrivals)
        arriving_stock = self.transit_queue.popleft()
        self.stocks += arriving_stock
        self.transit_queue.append(order_quantities)
        
        # 3. SIMULATE DEMAND
        demand = self.demand_df.iloc[self.current_day, 1:13].values.astype(np.float32)
        
        # 4. SALES & STOCKOUTS
        actual_sales = np.minimum(self.stocks, demand)
        stockouts = np.maximum(0, demand - self.stocks)
        
        # 5. REWARD FORMULA
        revenue = np.sum(actual_sales * self.unit_price)
        holding_expenses = np.sum(self.stocks * self.holding_cost)
        stockout_penalties = np.sum(stockouts * self.stockout_penalty)
        
        order_mask = order_quantities > 0
        ordering_expenses = np.sum(order_mask * self.order_fixed_cost) + \
                           np.sum(order_quantities * self.unit_cost)
        
        reward = revenue - holding_expenses - stockout_penalties - ordering_expenses
        
        # 6. UPDATE STATE
        self.stocks = np.clip(self.stocks - actual_sales, 0, 5000)
        self.current_day += 1
        done = self.current_day >= self.max_days
        
        return self._get_obs(), float(reward), done, False, {}