import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import timedelta
import csv
from sklearn.preprocessing import MinMaxScaler


class WeatherForecaster:
    def __init__(self, master_file=None, data=None, input_dim=3, hidden_dim=64, output_dim=1,
                 num_layers=2, learning_rate=0.001, batch_size=256, device=None, target_seq_length=1000):
        self.master_file = master_file
        self.data = data
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_seq_length = target_seq_length
        self.model = self.WeatherLSTM(input_dim, hidden_dim, output_dim, num_layers).to(self.device)
        self.criterion = nn.HuberLoss(delta=1.0)  # Huber Loss as default
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # Load and preprocess data
        master_data = self.data if self.data is not None else self.load_master_data()
        self.data = master_data[["DHT_Humidity_percent", "BMP_Temperature_C", "BMP_Pressure_hPa"]].values
        self.data = self.smooth_data(self.data, window_size=5)  # Smooth data
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)  # Normalize data
        self.seq_length = max(1, len(self.data) // self.target_seq_length)

    class WeatherLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])  # Take the last timestep output
            return out

    @staticmethod
    def smooth_data(data, window_size=5):
        """
        Apply a moving average to smooth the data.
        """
        smoothed_data = pd.DataFrame(data).rolling(window=window_size, min_periods=1, center=True).mean()
        return smoothed_data.values

    def save_model(self, model_path):
        """Save the model and optimizer state to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Load the model and optimizer state from a file."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.output_dim = checkpoint['output_dim']
        self.num_layers = checkpoint['num_layers']
        self.learning_rate = checkpoint['learning_rate']
        print(f"Model loaded from {model_path}")

    def load_master_data(self):
        """Load the master data from the CSV file."""
        data = pd.read_csv(self.master_file, on_bad_lines='skip')
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.dropna(subset=["Timestamp"])  # Drop rows with invalid timestamps
        data = data.sort_values("Timestamp").reset_index(drop=True)
        return data

    @staticmethod
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i+seq_length, :])  # Input sequence (seq_length rows, all columns)
            y.append(data[i+seq_length, 1])    # Target value (e.g., temperature at the next step)
        return np.array(x), np.array(y)

    def train_model(self, epochs=10, loss_csv_path="training_loss.csv", final_loss_csv_path="final_losses.csv"):
        x, y = self.create_sequences(self.data, self.seq_length)
        x_train = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        with open(loss_csv_path, mode='w', newline='') as loss_file:
            loss_writer = csv.writer(loss_file)
            loss_writer.writerow(["Epoch", "Loss"])

            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0

                for batch_x, batch_y in loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                epoch_loss /= len(loader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
                loss_writer.writerow([epoch + 1, epoch_loss])
                self.scheduler.step(epoch_loss)

        with open(final_loss_csv_path, mode='a', newline='') as final_loss_file:
            final_loss_writer = csv.writer(final_loss_file)
            final_loss_writer.writerow([epoch_loss])

    def predict_future(self, recent_sequence, steps_ahead=6):
        self.model.eval()
        future_predictions = []
        for _ in range(steps_ahead):
            with torch.no_grad():
                next_prediction = self.model(recent_sequence).item()
                future_predictions.append(next_prediction)
                last_timestep = recent_sequence[:, -1, :].clone()
                next_timestep = last_timestep.clone()
                next_timestep[0, 1] = next_prediction
                recent_sequence = torch.cat((recent_sequence[:, 1:, :], next_timestep.unsqueeze(1)), dim=1)
        return self.inverse_transform_predictions(future_predictions)

    def inverse_transform_predictions(self, predictions):
        """Reverse normalization for predictions."""
        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()

    @staticmethod
    def infer_timestamps(last_timestamp, steps_ahead, interval_seconds):
        future_timestamps = [
            last_timestamp + timedelta(seconds=interval_seconds * i)
            for i in range(1, steps_ahead + 1)
        ]
        return future_timestamps

    @staticmethod
    def save_predictions_to_csv(predictions, future_timestamps, output_file):
        df = pd.DataFrame({
            "Timestamp": future_timestamps,
            "Predicted_Temperature": predictions
        })
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

