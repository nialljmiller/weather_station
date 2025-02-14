import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import timedelta
import csv
from sklearn.preprocessing import MinMaxScaler



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import timedelta





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
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # Preprocessing logic
        master_data = self.data if self.data is not None else self.load_master_data()
        self.data = self.process_input_data(master_data[["DHT_Humidity_percent", "BMP_Temperature_C", "BMP_Pressure_hPa"]].values)
        
        

    def debug_data(self,data, step_name):
        """
        Debug the data for NaN and extreme values after each preprocessing step.
        :param data: Processed data as a NumPy array.
        :param step_name: Description of the current step.
        """
        print(f"Debugging data after {step_name}:")
        print(f"Shape: {data.shape}")
        print(f"Min: {np.nanmin(data)}")  # Handles NaN safely
        print(f"Max: {np.nanmax(data)}")
        print(f"NaN count: {np.isnan(data).sum()}")
        print(f"First row: {data[0]}")
        print(f"Last row: {data[-1]}")
        
        
        
    def process_input_data(self, data):
        data = self.validate_input_data(data)
        #self.debug_data(data, "raw input validation")

        data = self.smooth_data(data, window_size=10)
        #self.debug_data(data, "smoothing")

        # Hard scale temperature
        self.temp_min = np.min(data[:, 1])
        self.temp_max = np.max(data[:, 1])
        data[:, 1] = (data[:, 1] - self.temp_min) / (self.temp_max - self.temp_min) if self.temp_max != self.temp_min else 0
        #self.debug_data(data, "temperature scaling")

        # Hard scale humidity
        self.hum_min = np.min(data[:, 0])
        self.hum_max = np.max(data[:, 0])
        data[:, 0] = (data[:, 0] - self.hum_min) / (self.hum_max - self.hum_min) if self.hum_max != self.hum_min else 0
        #self.debug_data(data, "humidity scaling")

        # Hard scale pressure
        self.pres_min = np.min(data[:, 2])
        self.pres_max = np.max(data[:, 2])
        data[:, 2] = (data[:, 2] - self.pres_min) / (self.pres_max - self.pres_min) if self.pres_max != self.pres_min else 0
        #self.debug_data(data, "pressure scaling")

        # Add interaction terms
        humidity, temperature, pressure = data[:, 0], data[:, 1], data[:, 2]
        data = np.column_stack((data, humidity * temperature, humidity * pressure, temperature * pressure))
        #self.debug_data(data, "interaction terms")

        # Add lag features
        lags = [5, 30, 300]
        df = pd.DataFrame(data)
        for lag in lags:
            lagged_data = df.shift(lag).bfill().values
            data = np.column_stack((data, lagged_data))
        #self.debug_data(data, "lag features")

        # Add rate of change
        rate_of_change = np.diff(data, axis=0, prepend=data[0:1, :])
        data = np.column_stack((data, rate_of_change))
        #self.debug_data(data, "rate of change")

        # Final NaN cleaning
        if np.isnan(data).any():
            print("Warning: NaN detected in final processed data. Replacing with zeros...")
            data = np.nan_to_num(data, nan=0.0)

        # Update input_dim and seq_length
        self.input_dim = data.shape[1]
        self.seq_length = max(1, len(data) // self.target_seq_length)
        return data


    def validate_input_data(self, data):
        """
        Validate input data to ensure there are no NaN or extreme values.
        :param data: Input data as a NumPy array.
        :return: Cleaned data with issues logged.
        """
        if np.isnan(data).any():
            print("Warning: Input data contains NaN values. Attempting to fill...")
            data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
        if np.isinf(data).any():
            print("Warning: Input data contains infinite values. Clipping...")
            data = np.clip(data, a_min=np.finfo(np.float32).min, a_max=np.finfo(np.float32).max)
        return data


    def inverse_transform_predictions(self, predictions):
        """
        Reverse scaling for temperature predictions.
        :param predictions: Scaled temperature predictions.
        :return: Original scale predictions.
        """
        return predictions * (self.temp_max - self.temp_min) + self.temp_min

    def predict_future(self, recent_sequence, steps_ahead=6):
        """
        Predict future temperature values.
        :param recent_sequence: Recent sequence of normalized input data (3D tensor).
        :param steps_ahead: Number of future steps to predict.
        :return: Denormalized future predictions.
        """
        self.model.eval()
        future_predictions = []

        # Convert recent_sequence to 2D to process
        if isinstance(recent_sequence, torch.Tensor):
            recent_sequence = recent_sequence.squeeze(0).cpu().numpy()  # Shape: (seq_length, input_dim)

        # Preprocess input sequence
        processed_sequence = self.process_input_data(recent_sequence)

        # Convert back to 3D tensor for LSTM
        processed_sequence = torch.tensor(processed_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, seq_length, input_dim)


        # Debugging processed_sequence
        #print("Processed sequence stats:")
        #print(f"Shape: {processed_sequence.shape}")
        #print(f"Min: {processed_sequence.min().item()}")
        #print(f"Max: {processed_sequence.max().item()}")
        #print(f"NaN count: {(processed_sequence != processed_sequence).sum().item()}")

        # Handle NaN values
        if torch.isnan(processed_sequence).any():
            print("Warning: Processed sequence contains NaN values. Attempting to fix...")
            processed_sequence = torch.nan_to_num(processed_sequence, nan=0.0)

        for _ in range(steps_ahead):
            with torch.no_grad():
                # Predict the next value
                next_prediction = self.model(processed_sequence).item()
                future_predictions.append(next_prediction)

                # Update the sequence with the predicted value
                last_timestep = processed_sequence[:, -1, :].clone()
                next_timestep = last_timestep.clone()
                next_timestep[0, 1] = next_prediction # Update temperature feature

                # Shift sequence and append the new timestep
                processed_sequence = torch.cat((processed_sequence[:, 1:, :], next_timestep.unsqueeze(1)), dim=1)

        # Denormalize predictions
        return self.inverse_transform_predictions(np.array(future_predictions))


    @staticmethod
    def smooth_data(data, window_size=10):
        """
        Apply a moving average to smooth the data.
        :param data: Original data.
        :param window_size: Window size for smoothing.
        :return: Smoothed data.
        """
        smoothed_data = pd.DataFrame(data).rolling(window=window_size, min_periods=1, center=True).mean()
        return smoothed_data.values


    def load_master_data(self):
        """Load the master data from the CSV file."""
        data = pd.read_csv(self.master_file, on_bad_lines='skip')
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.dropna(subset=["Timestamp"])  # Drop rows with invalid timestamps
        data = data.sort_values("Timestamp").reset_index(drop=True)
        return data

    class WeatherLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])  # Take the last timestep output
            return out


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
        """
        Load the model and optimizer state from a file.
        :param model_path: Path to load the model from.
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.output_dim = checkpoint['output_dim']
        self.num_layers = checkpoint['num_layers']
        self.learning_rate = checkpoint['learning_rate']
        print(f"Model loaded from {model_path}")


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
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.10f}")
                loss_writer.writerow([epoch + 1, epoch_loss])
                self.scheduler.step(epoch_loss)

        with open(final_loss_csv_path, mode='a', newline='') as final_loss_file:
            final_loss_writer = csv.writer(final_loss_file)
            final_loss_writer.writerow([epoch_loss])


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

