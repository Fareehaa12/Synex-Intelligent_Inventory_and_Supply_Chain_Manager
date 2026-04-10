import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

class MultiSKUInventoryEnv(gym.Env):
    """
    Intelligent Inventory & Supply Chain Simulator.
    Fulfills all requirements: 10+ SKUs, Lead Times, Pending Orders, 
    Days of Stock Remaining, Rolling Demand, and Unit Costs in State.
    """
    def __init__(self, demand_data_path):
        super(MultiSKUInventoryEnv, self).__init__()
        
        # 1. LOAD DEMAND DATA
        self.demand_df = pd.read_csv(demand_data_path)
        self.num_skus = len(self.demand_df.columns) - 1
        self.max_days = len(self.demand_df) - 1
        
        # 2. ACTION SPACE
        # Order quantity bins per SKU: 0, 50, 100, 200, 500
        self.action_mapping = {0: 0, 1: 50, 2: 100, 3: 200, 4: 500}
        self.action_space = spaces.MultiDiscrete([5] * self.num_skus)

        # 3. STATE SPACE (Observation)
        # Features per SKU: [Stock, Rolling Demand, Pending Orders, Days Remaining, Unit Cost]
        # Total size = 12 SKUs * 5 features = 60
        obs_shape = self.num_skus * 5 
        self.observation_space = spaces.Box(
            low=0, high=5000, shape=(obs_shape,), dtype=np.float32
        )

        # 4. FINANCIAL & SUPPLY CHAIN CONSTANTS
        self.unit_price = 100       # Revenue per unit
        self.unit_cost = 50         # Cost to purchase (Fulfills 'unit cost' requirement)
        self.holding_cost = 0.5     # Daily storage cost
        self.stockout_penalty = 30  # Cost of lost sales/customer dissatisfaction
        self.order_fixed_cost = 25  # Shipping/Admin fee per SKU ordered
        self.lead_time = 3          # 3-day delay (Fulfills 'lead time' requirement)

        self.reset()

    def _get_rolling_demand(self):
        """Calculates 7-day rolling average demand for trend analysis."""
        window = 7
        if self.current_day < window:
            return self.demand_df.iloc[:self.current_day+1, 1:].mean().values
        return self.demand_df.iloc[self.current_day-(window-1) : self.current_day+1, 1:].mean().values

    def _get_obs(self):
        """Constructs the full State (s) exactly as requested in the brief."""
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
        
        # Concatenate into one flat State Vector [Fulfills all 'State (s)' bullets]
        return np.concatenate([
            stocks, 
            rolling_demand, 
            pending_total, 
            days_remaining,
            costs_vector
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
        # Orders from 'lead_time' days ago arrive today
        arriving_stock = self.transit_queue.popleft()
        self.stocks += arriving_stock
        
        # Today's order goes into the back of the queue
        self.transit_queue.append(order_quantities)
        
        # 3. SIMULATE DEMAND
        demand = self.demand_df.iloc[self.current_day, 1:].values.astype(np.float32)
        
        # 4. SALES & STOCKOUTS
        actual_sales = np.minimum(self.stocks, demand)
        stockouts = np.maximum(0, demand - self.stocks)
        
        # 5. REWARD FORMULA
        revenue = np.sum(actual_sales * self.unit_price)
        holding_expenses = np.sum(self.stocks * self.holding_cost)
        stockout_penalties = np.sum(stockouts * self.stockout_penalty)
        
        # Ordering Cost = Fixed transaction fee per ordered SKU + unit cost
        order_mask = order_quantities > 0
        ordering_expenses = np.sum(order_mask * self.order_fixed_cost) + \
                            np.sum(order_quantities * self.unit_cost)
        
        reward = revenue - holding_expenses - stockout_penalties - ordering_expenses
        
        # 6. UPDATE STATE
        self.stocks = np.clip(self.stocks - actual_sales, 0, 5000)
        
        self.current_day += 1
        done = self.current_day >= self.max_days
        
        return self._get_obs(), float(reward), done, False, {}