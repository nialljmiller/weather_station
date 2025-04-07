# Weather Station

This repository contains a comprehensive weather monitoring and prediction system built using a Raspberry Pi and several environmental sensors. The project collects, processes, and visualizes weather data while incorporating predictive modeling for future temperature trends.

## Features

- **Real-Time Data Collection**: Uses multiple sensors to collect data every 10 seconds.
- **Data Visualization**: Generates real-time plots to display trends in temperature, humidity, pressure, and light intensity.
- **Predictive Modeling**: Utilizes an LSTM-based recurrent neural network (RNN) to predict future temperature values.
- **System Metrics Monitoring**: Tracks the performance of the Raspberry Pi, including CPU and memory usage.

---

## Sensors Used

### Barometric Pressure Sensor (BMP085)
- **Measures**: Air pressure and altitude.
- **Usage**: Detect weather changes; calculate altitude above sea level.

### Temperature and Humidity Sensor (DHT11)
- **Measures**: Air temperature and humidity.
- **Usage**: Assess comfort levels and provide additional context to weather data.

### Light Sensor (BH1750)
- **Measures**: Light intensity in lux.
- **Usage**: Determine environmental brightness based on the time of day and weather conditions.

---

## Workflow

1. **Data Collection**: Sensors collect temperature, humidity, pressure, and light data every 10 seconds.
2. **Data Transmission**: The Raspberry Pi sends data to the server every 30 seconds.
3. **Data Processing**: 
    - Server processes data into visualizations every minute.
    - Rolling averages and summaries are generated periodically.
4. **Machine Learning**: An RNN model retrains daily using updated data to predict future temperatures.
5. **Visualization**: The system generates plots and dashboards for real-time monitoring and insights.

---

## Key Scripts

- **`weather_station.py`**:
  - Interfaces with sensors to gather environmental data.
  - Logs data locally and transfers it to a central server.

- **`server_weather_processing.py`**:
  - Processes incoming data from the Raspberry Pi.
  - Generates rolling averages, summary plots, and additional metrics.

- **`weather_forcast.py`**:
  - Implements a Long Short-Term Memory (LSTM) network for temperature prediction.
  - Handles training, prediction, and model saving/loading.

---

## Plots and Dashboards

The system generates multiple types of plots:
- **Temperature, Humidity, Pressure, and Light Trends**: Display changes over time.
- **Heat Index and Environmental Comfort Index (ECI)**: Assess environmental livability.
- **Dew and Frost Points**: Indicate condensation and freezing risks.

Rolling averages and summaries are exported in CSV and HTML formats.

---

## Usage

### Hardware Requirements
- Raspberry Pi (any recent model with sufficient processing power).
- BMP085, DHT11, and BH1750 sensors.
- Compatible I2C and GPIO connections for the sensors.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nialljmiller/weather_station.git
   cd weather_station
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
1. Set up sensors and connect them to the Raspberry Pi.
2. Start the data collection script:
   ```bash
   python weather_station.py
   ```
3. Ensure the server processing script is running to visualize and predict data:
   ```bash
   python server_weather_processing.py
   ```
