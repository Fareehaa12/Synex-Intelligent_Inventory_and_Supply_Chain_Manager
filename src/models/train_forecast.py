import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# --- Configuration ---
LOOKBACK = 7  # Use last 7 days to predict
PREDICT_FORWARD = 3  # Predict next 3 days
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 60
BATCH_SIZE = 32
SAVE_DIR = "saved_models/demand_forecast_lstm"

class DemandLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, output_size=36):
        super(DemandLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, lookback, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(df):
    scaler = MinMaxScaler()
    
    # Check if 'Day' column exists and drop it to keep only the 12 SKUs
    if 'Day' in df.columns:
        print("📊 'Day' column detected. Removing it to focus on SKU demand...")
        df = df.drop(columns=['Day'])
    elif df.shape[1] > 12:
        # Fallback: if the column isn't named 'Day' but there's an extra, take the last 12
        print(f"⚠️ Found {df.shape[1]} columns. Trimming to the last 12 SKUs...")
        df = df.iloc[:, -12:]
        
    data = df.values
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - LOOKBACK - PREDICT_FORWARD + 1):
        X.append(scaled_data[i : i + LOOKBACK])
        # Target is the next 3 days flattened (36 values)
        y.append(scaled_data[i + LOOKBACK : i + LOOKBACK + PREDICT_FORWARD].flatten())
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler

def train_forecaster():
    # Create the professional subfolder structure
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load data from the data folder
    if not os.path.exists("data/demand_history.csv"):
        print("❌ Error: data/demand_history.csv not found!")
        return

    df = pd.read_csv("data/demand_history.csv")
    
    # Pre-processing
    X, y, scaler = prepare_data(df)
    
    model = DemandLSTM(input_size=12, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=36)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"🚀 Training Ultimate Forecaster on {len(X)} samples...")
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"📅 Epoch [{epoch+1}/{EPOCHS}] | Loss: {loss.item():.8f}")

    # --- Save Assets ---
    model_path = os.path.join(SAVE_DIR, "demand_lstm.pth")
    scaler_path = os.path.join(SAVE_DIR, "demand_scaler.pkl")
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    print("-" * 30)
    print(f"✅ SUCCESS!")
    print(f"📍 Model: {model_path}")
    print(f"📍 Scaler: {scaler_path}")
    print("-" * 30)

if __name__ == "__main__":
    train_forecaster()