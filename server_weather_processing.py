import os
import csv
import json
import time
import warnings
from datetime import datetime, timedelta


import numpy as np
import pandas as pd
import pytz
from pytz import UTC  # Ensure all datetimes are timezone-aware
import torch
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from numpy.polynomial import Polynomial
import GPy
import mimetypes
from matplotlib.dates import DateFormatter

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections
import matplotlib.dates
import matplotlib.ticker as mticker

from weather_forcast import WeatherForecaster  # Assuming it's a custom module

# Suppress specific warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        
import psutil
from pynvml import *
import subprocess
import csv
from datetime import datetime


import psutil
import csv
from datetime import datetime
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlDeviceGetTemperature, nvmlShutdown
import subprocess


def initialize_csv(output_file="system_stats.csv"):
    """Initializes the CSV file with the header."""
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Timestamp",
            "CPU Usage (%)",
            "CPU Temp (°C)",
            "Memory Usage (%)",
            "GPU Usage (%)",
            "GPU Memory Usage (%)",
            "GPU Temp (°C)",
            "Disk Usage",
            "Net Disk I/O (MB)",
            "Thermals"
        ])
        
        

def gather_system_stats(output_file="system_stats.csv"):
    """Appends detailed system stats to the CSV file."""
    # Initialize NVIDIA Management Library (for GPU stats)
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assuming a single NVIDIA GPU

    # Gather CPU usage and temperature
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_temp = None
    try:
        # Attempt to get CPU temperature (Linux-specific)
        cpu_temp = psutil.sensors_temperatures()["coretemp"][0].current
    except KeyError:
        cpu_temp = "N/A"

    # Gather memory usage stats
    memory = psutil.virtual_memory()
    memory_usage = memory.percent

    # Gather GPU usage and temperature
    gpu_utilization = nvmlDeviceGetUtilizationRates(gpu_handle)
    gpu_memory = nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_temp = nvmlDeviceGetTemperature(gpu_handle, NVML_TEMPERATURE_GPU)

    # Gather disk usage stats
    disk_usage = []
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_usage.append({
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "used": usage.used,
                "total": usage.total,
                "percent": usage.percent
            })
        except PermissionError:
            continue

    # Gather net disk I/O
    disk_io = psutil.disk_io_counters()
    total_read = disk_io.read_bytes / (1024**2)  # Convert to MB
    total_write = disk_io.write_bytes / (1024**2)  # Convert to MB

    # Gather additional temperatures using `sensors`
    formatted_temps = {}
    try:
        sensors_output = subprocess.check_output(["sensors"], text=True).splitlines()
        for line in sensors_output:
            if ":" in line:
                parts = line.split(":")
                label = parts[0].strip()
                temp_data = parts[1].strip().split()
                if temp_data and temp_data[0].replace(".", "").isdigit():
                    formatted_temps[label] = float(temp_data[0].replace("°C", ""))
    except Exception as e:
        print(f"Error fetching additional temperatures: {e}")

    # Format disk usage and temperatures for CSV
    disk_usage_str = "; ".join(
        f"{entry['device']}({entry['mountpoint']}): {entry['used']/1024**3:.2f}GB/{entry['total']/1024**3:.2f}GB ({entry['percent']}%)"
        for entry in disk_usage
    )
    temp_str = "; ".join(f"{key}: {value}°C" for key, value in formatted_temps.items())

    # Append data to CSV
    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now(),
            f"{cpu_usage}%",
            f"{cpu_temp}°C" if cpu_temp != "N/A" else "N/A",
            f"{memory_usage}%",
            f"{gpu_utilization.gpu}%",
            f"{gpu_memory.used / gpu_memory.total * 100:.2f}%",
            f"{gpu_temp}°C",
            disk_usage_str,
            f"Read: {total_read:.2f}MB, Write: {total_write:.2f}MB",
            temp_str
        ])

    nvmlShutdown()




def plot_training_loss(file_path="training_loss.csv", output_path="training_loss_plot.png"):
    """
    Reads training_loss.csv and saves a line plot of the loss per epoch.
    The plot includes the file creation date in the title and uses an exponential scale for the loss axis.
    """
    try:
        # Get file creation date
        creation_date = None
        if os.path.exists(file_path):
            creation_timestamp = os.path.getmtime(file_path)
            creation_date = datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d')

        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            data = list(reader)

            if len(data) < 2:
                print("No training data found in the file to plot.")
                return

            # Extracting data
            epochs = []
            losses = []
            for row in data[1:]:  # Skip header
                epochs.append(int(row[0]))
                losses.append(float(row[1]))

            # Plotting
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, losses, marker='o', linestyle='-', label='Loss')
            title = "Training Loss Per Epoch"
            if creation_date:
                title += f" (File Created: {creation_date})"
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale('log')  # Set y-axis to exponential (logarithmic) scale
            plt.grid(True, which="both", linestyle='--', linewidth=0.5)
            plt.legend()
            plt.savefig(output_path)
            plt.close()

            print(f"Training loss plot saved to {output_path}.")

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_final_losses(file_path="final_losses.csv", output_path="final_losses_plot.png"):
    """
    Reads final_losses.csv and saves a bar plot of the final losses across runs.
    """
    try:
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            data = list(reader)

            if not data:
                print("No final losses data found in the file to plot.")
                return

            # Extracting data
            runs = list(range(1, len(data) + 1))
            losses = [float(row[0]) for row in data]

            # Plotting
            plt.figure(figsize=(8, 6))
            plt.bar(runs, losses, color='blue', alpha=0.7, label='Final Loss')
            plt.title("Final Losses Across Runs")
            plt.xlabel("Run")
            plt.ylabel("Loss")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(runs)
            plt.legend()
            plt.savefig(output_path)
            plt.close()

            print(f"Final losses plot saved to {output_path}.")

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



# File paths
INCOMING_FILE = "/media/bigdata/weather_station/weather_data.csv"
MASTER_FILE = "/media/bigdata/weather_station/all_data.csv"
MASTER_FILE_json = "/media/bigdata/weather_station/all_data.json"
PLOT_ALL_TIME = "/media/bigdata/weather_station/weather_plot_all.png"
PLOT_1_DAY = "/media/bigdata/weather_station/weather_plot_1_day.png"
PLOT_1_HOUR = "/media/bigdata/weather_station/weather_plot_1_hour.png"
ROLLING_AVERAGES_FILE = "/media/bigdata/weather_station/rolling_averages.csv"
PREDICT_FILE = "/media/bigdata/weather_station/predictions.csv"


# Ensure master file exists
if not os.path.exists(MASTER_FILE):
    pd.DataFrame(columns=[
        "Timestamp", "BMP_Temperature_C", "BMP_Pressure_hPa",
        "BMP_Altitude_m", "DHT_Temperature_C", "DHT_Humidity_percent",
        "BH1750_Light_lx"
    ]).to_csv(MASTER_FILE, index=False)

def load_master_data(fp):
    """Load the master data from the CSV file."""
    data = pd.read_csv(fp, on_bad_lines='skip') 
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data = data.dropna(subset=["Timestamp"])  # Drop rows with invalid timestamps
    data = data.sort_values("Timestamp").reset_index(drop=True)
    return data




def generate_json_from_csv(csv_path, json_path):
    """
    Convert CSV data to JSON format.

    Args:
        csv_path (str): Path to the source CSV file.
        json_path (str): Path to save the generated JSON file.
    """
    try:
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)  # Reads CSV as dictionaries
            data = list(reader)  # Convert the entire CSV into a list of dictionaries

        # Save the data to a JSON file
        with open(json_path, "w") as jsonfile:
            json.dump(data, jsonfile, indent=4)

        print(f"JSON data successfully written to {json_path}")

    except Exception as e:
        print(f"Error generating JSON file: {e}")


def append_new_data(master_data):
    """Append new data from the incoming file to the master DataFrame."""
    if not os.path.exists(INCOMING_FILE):
        print(f"Incoming file {INCOMING_FILE} does not exist. Skipping this iteration.")
        return master_data

    # Validate file type
    file_type, encoding = mimetypes.guess_type(INCOMING_FILE)
    if file_type != "text/csv":
        print(f"Warning: {INCOMING_FILE} is not a text/csv file. Detected type: {file_type}")
        return master_data

    # Load incoming data
    try:
        incoming_data = pd.read_csv(INCOMING_FILE, encoding="utf-8", on_bad_lines="skip")
        incoming_data["Timestamp"] = pd.to_datetime(incoming_data["Timestamp"], errors="coerce")
        incoming_data = incoming_data.dropna(subset=["Timestamp"])
    except Exception as e:
        print(f"Error loading incoming data: {e}")
        return master_data

    # Concatenate and remove duplicates
    combined_data = pd.concat([master_data, incoming_data], ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset="Timestamp").sort_values("Timestamp").reset_index(drop=True)

    # Save to master file
    combined_data.to_csv(MASTER_FILE, index=False)
    generate_json_from_csv(MASTER_FILE, MASTER_FILE_json)
    print(f"Appended new data and saved to {MASTER_FILE}. Total rows: {len(combined_data)}.")

    return combined_data



def generate_summary_plot(data, output_path):
    """Generate a single-panel summary plot for smoothed temperature and humidity."""
    fig, ax_temp_c = plt.subplots(figsize=(10, 6))

    # Plot smoothed temperature (°C) on the primary y-axis (left-hand side)
    ax_temp_c.plot(data["Timestamp"], data["Median_Temperature_C"], color="purple", alpha=0.7, label="Temperature")
    ax_temp_c.tick_params(axis="y", labelcolor="blue")

    # Add Fahrenheit scale on a secondary left y-axis (stacked with °C)
    ax_temp_f = ax_temp_c.twinx()
    ax_temp_f.spines["left"].set_position(("axes", 0))#-0.15))  # Offset the Fahrenheit axis
    ax_temp_f.plot(data["Timestamp"], data["Median_Temperature_C"] * 9 / 5 + 32, color="purple", alpha=0.7, label="Temperature (°F)")
    ax_temp_f.set_ylabel("Temperature (°C/°F)", color="purple")
    ax_temp_f.tick_params(axis="y", labelcolor="red")
    ax_temp_f.yaxis.set_label_position("left")
    ax_temp_f.yaxis.tick_left()

    # Plot smoothed humidity on the right-hand side y-axis
    ax_hum = ax_temp_c.twinx()
    ax_hum.spines["right"].set_visible(True)
    ax_hum.plot(data["Timestamp"], data["DHT_Humidity_percent_Smoothed"], label="Humidity (%)", color="green", alpha=0.7)
    ax_hum.set_ylabel("Humidity (%)", color="green")
    ax_hum.tick_params(axis="y", labelcolor="green")

    # Title and legend
    ax_temp_c.legend(loc="upper left")
    ax_hum.legend(loc="upper right")

    # Formatting
    ax_temp_c.xaxis.set_tick_params(rotation=45)
    ax_temp_c.grid(alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\tSaved summary plot to {output_path}.")


def generate_plots(data, predict_data, output_path, title):
    """Generate a 4x4 subplot for temperature, humidity, pressure, and light with additional calculated metrics."""
    # Convert timestamps to Mountain Time
    mountain_tz = pytz.timezone("America/Denver")
    data["Timestamp"] = data["Timestamp"].dt.tz_convert(mountain_tz)
    predict_data["Timestamp"] = predict_data["Timestamp"].dt.tz_convert(mountain_tz)
    altitude_m = data["BMP_Altitude_m"]
    # Calculate median temperature

    # Replace NaNs and Infs in all numeric columns
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Convert temperatures to Fahrenheit
    data["BMP_Temperature_F"] = data["BMP_Temperature_C"] * 9 / 5 + 32
    data["DHT_Temperature_F"] = data["DHT_Temperature_C"] * 9 / 5 + 32


    # Ensure the Timestamp column is clean and valid
    if "Timestamp" in data.columns:
        data = data.dropna(subset=["Timestamp"])  # Drop rows with missing timestamps
        if not pd.api.types.is_datetime64_any_dtype(data["Timestamp"]):
            # Convert to datetime if not already in the correct format
            data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.dropna(subset=["Timestamp"])  # Drop rows with invalid timestamps


    timestamps = np.arange(len(data["Timestamp"]))
    smooth_humidity = data["DHT_Humidity_percent_Smoothed"].values




    # Heat Index Calculation
    T = data["DHT_Temperature_F"]
    H = smooth_humidity#Humidity as a percentage

    h = altitude_m
  
    data["Heat_Index"] = np.where(
        data["Median_Temperature_C"] > 27,  # Use heat index for temps >27°C
        -42.379 + 2.04901523 * data["DHT_Temperature_F"] + 10.14333127 * H
        - 0.22475541 * data["DHT_Temperature_F"] * H
        - 0.00683783 * data["DHT_Temperature_F"]**2
        - 0.05481717 * H**2
        + 0.00122874 * data["DHT_Temperature_F"]**2 * H
        + 0.00085282 * data["DHT_Temperature_F"] * H**2
        - 0.00000199 * data["DHT_Temperature_F"]**2 * H**2,
        data["Median_Temperature_C"],  # Below threshold, return actual temperature
    )
              


    # Use the Magnus-Tetens formula for vapor pressure
    e = H / 100 * 6.112 * np.exp((17.62 * data["Median_Temperature_C"]) / (data["Median_Temperature_C"] + 243.12))

    # Calculate specific humidity
    data["Specific_Humidity_gkg"] = 0.622 * e / (data["Sea_Level_Pressure_hPa"] - e) * 1000





    # Normalize factors
    H_norm = H / 100  # Humidity as a fraction
    T_comfort = np.clip((data["Median_Temperature_C"] - 21) / 6, -1, 1)  # Centered at 21°C for comfort zone
    T_norm = 1 - abs(T_comfort)  # Invert so values closer to 21°C score higher
    P_norm = np.clip((data["BMP_Pressure_hPa"] - 1013) / 50, -1, 1)  # Pressure normalized, with a cap
    L_norm = np.clip(data["BH1750_Light_lx"] / 50000, 0, 1)  # Ambient light normalized (up to 50,000 lx)

    # Weight factors for impact
    ECI = (
        0.3 * H_norm +  # Humidity
        0.5 * T_norm +  # Temperature
        0.1 * P_norm +  # Pressure
        0.2 * L_norm    # Ambient Light
    )

    # Scale the result to 0-1 range
    data["ECI"] = np.clip(ECI, 0, 1)
    
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    

    # Calculate the maximum length allowed for predicted data (1/4th of the main data length)
    max_predicted_length = len(data["Timestamp"]) // 4

    # Find the starting index of the predictions after the real data
    start_index = predict_data["Timestamp"].searchsorted(data["Timestamp"].iloc[-1], side="right")

    # Ensure predicted data starts immediately after the real data and is limited to the specified length
    predict_data_subset = predict_data.iloc[start_index:start_index + max_predicted_length]

    # Find overlapping predicted data (predictions before the end of the real data)
    overlap_index = predict_data["Timestamp"].searchsorted(data["Timestamp"].iloc[-1], side="left")
    predict_data_overlap = predict_data.iloc[:overlap_index]

    # Temperature Plot
    ax1 = axs[0, 0]
    ax1.plot(data["Timestamp"], data["BMP_Temperature_C"], color="blue", alpha=0.1)
    ax1.plot(data["Timestamp"], data["DHT_Temperature_C"], color="cyan", alpha=0.1)
    ax1.plot(data["Timestamp"], data["BMP_Temperature_Smoothed"], label="BMP Temp", color="blue", alpha=0.7)
    ax1.plot(data["Timestamp"], data["DHT_Temperature_Smoothed"], label="DHT Temp", color="cyan", alpha=0.7)    
    ax1.plot(predict_data_subset["Timestamp"], predict_data_subset["Predicted_Temperature"], label="Predicted Temp", color="royalblue", alpha=0.7)
    ax1.plot(data["Timestamp"], data["Median_Temperature_C"], color="magenta", linestyle="--")
    ax1.set_title("Temperature")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper left")
    ax1.grid()

    
    # Add Fahrenheit Scale
    ax_f = ax1.twinx()
    ax_f.plot(data["Timestamp"], data["BMP_Temperature_C"] * 9/5 + 32, alpha=0)  # Invisible line for correct scaling
    ax_f.set_ylabel("Temperature (°F)", color="red")
    ax_f.tick_params(axis="y", labelcolor="red")


    # Humidity Plot with Cubic Spline
    ax2 = axs[0, 1]
    ax2.plot(data["Timestamp"], data["DHT_Humidity_percent"], color="green", alpha=0.01)
    ax2.plot(data["Timestamp"], data["DHT_Humidity_percent_Smoothed"], color="green",  label="Humidity (%)", alpha=0.7)

    ax2.set_title("Humidity")
    ax2.set_ylabel("Humidity (%)")
    ax2.legend()
    ax2.grid()
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    


    # Combined Pressure and Sea-Level Pressure Plot
    ax3 = axs[1, 0]  # Use the left plot in the second row
    ax3_secondary = ax3.twinx()

    # Adjust the rendering order of the twin axes
    ax3_secondary.set_zorder(ax3.get_zorder() - 1)
    ax3.set_facecolor((0, 0, 0, 0))  # Make ax3 background transparent

    # Plot Sea-Level Pressure first (green lines)
    ax3_secondary.plot(data["Timestamp"], data["Sea_Level_Pressure_hPa"]/10, label="Sea-Level Pressure (kPa)", color="green", alpha=0.3, zorder=1)
    ax3_secondary.plot(data["Timestamp"], data["Sea_Level_Pressure_hPa_Smoothed"]/10, color="green", zorder=1)
    ax3_secondary.set_ylabel("Sea-Level Pressure (kPa)", color="green")
    ax3_secondary.tick_params(axis="y", labelcolor="green")
    ax3_secondary.legend(loc="upper right")
    ax3_secondary.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.3f}"))  # Formats numbers to 3 decimal places

    # Plot Pressure on top (red lines)
    ax3.plot(data["Timestamp"], data["BMP_Pressure_hPa"]/10, label="Pressure (kPa)", color="red", alpha=0.3, zorder=2)
    ax3.plot(data["Timestamp"], data["BMP_Pressure_hPa_Smoothed"]/10, color="red", zorder=3)

    ax3.set_title("Pressure and Sea-Level Pressure")
    ax3.set_ylabel("Pressure (kPa)", color="red")
    ax3.tick_params(axis="y", labelcolor="red")
    ax3.legend(loc="upper left")
    ax3.grid()

    # Light Plot
    axs[1, 1].plot(data["Timestamp"], data["BH1750_Light_lx"], label="Light (lx)", color="orange")
    axs[1, 1].plot(data["Timestamp"], data["BH1750_Light_lx_Smoothed"], color="orange", alpha=0.3)
    
    axs[1, 1].set_title("Light")
    axs[1, 1].set_ylabel("Light (lx)")
    axs[1, 1].legend()
    axs[1, 1].grid()

   
    # Heat Index Plot
    ax2 = axs[2, 0]
    ax2.plot(data["Timestamp"], data["Heat_Index"], label="Heat Index", color="orange")
    ax2.set_title("Heat Index")
    ax2.set_ylabel("Heat Index")
    ax2.legend()
    ax2.grid()
    
    
    
    
    # Dew Point Plot
    ax3 = axs[2, 1]
    
        
        # Function to apply a buffer to transitions
    def apply_buffer(data, column_name, threshold=0):
        values = data[column_name].values
        buffered_values = values.copy()
        
        for i in range(1, len(values) - 1):
            # Check for crossing into negative (or positive)
            if (values[i] >= threshold and values[i - 1] < threshold) or \
               (values[i] < threshold and values[i - 1] >= threshold):
                # Apply buffer by extending the previous value
                buffered_values[i + 1] = values[i]
                buffered_values[i + 2] = values[i+1]
        
        return buffered_values

    # Create separate series with NaNs to avoid connecting lines
    dew_point_above = data["Dew_Point_C_smoothed"].where(data["Dew_Point_C_smoothed"] >= 0)  # Keep dew points above 0
    dew_point_below = data["Dew_Point_C_smoothed"].where(data["Dew_Point_C_smoothed"] < 0)   # Keep frost points below 0

# Apply buffer to dew point
    data["Dew_Point_C_Buffered"] = apply_buffer(data, "Dew_Point_C", threshold=0)

    # Plot Dew Point (above freezing) in blue
    ax3.plot(
        data["Timestamp"],
        dew_point_above,
        label="Dew Point (°C)",
        color="blue",
    )

    # Plot Frost Point (below freezing) in light blue
    ax3.plot(
        data["Timestamp"],
        dew_point_below,
        label="Frost Point (°C)",
        color="lightblue",
        alpha=0.7,
    )

    # Set plot title, labels, and legend
    ax3.set_title("Dew and Frost Point")
    ax3.set_ylabel("Point Temperature (°C)")
    ax3.legend()
    ax3.grid()

    # Calculate combined min and max for consistent y-axis scaling
    combined_min = data["Dew_Point_C"].min(skipna=True)
    combined_max = data["Dew_Point_C"].max(skipna=True)

    # Validate the calculated limits
    if pd.isna(combined_min) or pd.isna(combined_max):
        print("Warning: Dew Point data contains invalid values. Setting default axis limits.")
        combined_min, combined_max = -10, 40  # Default range (customize as needed)

    # Add margins to the limits
    if combined_min < 0:
        combined_min *= 1.1
    else:
        combined_min *= 0.9

    if combined_max < 0:
        combined_max *= 0.8
    else:
        combined_max *= 1.1

    # Safeguard: Ensure valid axis limits are applied
    if combined_min >= combined_max:
        print("Warning: Invalid axis limits calculated. Setting fallback limits.")
        combined_min, combined_max = -10, 40  # Default range

    ax3.set_ylim(combined_min, combined_max)

    # Specific Humidity Plot
    ax5 = axs[3, 0]
    ax5.plot(data["Timestamp"], data["Specific_Humidity_gkg"], label="Specific Humidity (g/kg)", color="brown")
    ax5.set_title("Specific Humidity")
    ax5.set_ylabel("Specific Humidity (g/kg)")
    ax5.legend()
    ax5.grid()
    
    # Environmental Comfort Index Plot
    ax6 = axs[3, 1]
    ax6.plot(data["Timestamp"], data["ECI"], color="k", linewidth=1.5)
    # Normalize ECI values for color mapping
    norm = plt.Normalize(0,1)#data["ECI"].min(), data["ECI"].max())
    cmap = plt.cm.get_cmap("RdYlGn")  # Red (discomfort) to Green (comfort)

    # Convert timestamps to numerical format for plotting
    timestamps_num = matplotlib.dates.date2num(data["Timestamp"])

    # Create a color-mapped line
    points = np.array([timestamps_num, data["ECI"]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = matplotlib.collections.LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(data["ECI"])
    lc.set_linewidth(1)

    # Add the line collection to the axis
    line = ax6.add_collection(lc)

    # Add color bar for reference
    cbar = plt.colorbar(line, ax=ax6, orientation="vertical", pad=0.02)
    cbar.set_label("Environmental Comfort Index (ECI)")





    # Axis formatting
    ax6.set_title("Environmental Comfort Index")
    ax6.set_ylabel("ECI")
    ax6.grid()

    # Formatting
    for ax in axs.flat:
        ax.xaxis.set_tick_params(rotation=45)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for the title
    plt.savefig(output_path)
    plt.close()
    ax1.ticklabel_format(style="plain", axis="y")
    ax2.ticklabel_format(style="plain", axis="y")
    ax3.ticklabel_format(style="plain", axis="y")
    ax5.ticklabel_format(style="plain", axis="y")
    ax6.ticklabel_format(style="plain", axis="y")
    ax1.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    ax2.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    ax3.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    ax5.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    ax6.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    print(f"\tSaved plot to {output_path}.")


def save_last_minute_averages(data, predict_data, output_file):
    """Save the last 1-minute averages for temperature, humidity, pressure, and light as an HTML file."""
    # Filter data for the last 1 minute
    last_minute_data = data[data["Timestamp"] >= (data["Timestamp"].max() - pd.Timedelta(minutes=1))]

    # Calculate averages
    averages = {
        "Temperature (°C/°F)": f"{last_minute_data['Median_Temperature_C'].mean():.2f}°C / {last_minute_data['Median_Temperature_F'].mean():.2f}°F",
        "Predicted Temp (Hour avg) (°C/°F)": f"{predict_data['Predicted_Temperature'].mean():.2f}°C / {predict_data['Predicted_Temperature'].mean() * 9 / 5 + 32:.2f}°F",
        "Humidity (%)": last_minute_data["DHT_Humidity_percent"].mean(),
        "Pressure (kPa)": last_minute_data["BMP_Pressure_hPa"].mean() / 10,
        "Light (lx)": last_minute_data["BH1750_Light_lx"].mean(),
    }

    # Create a DataFrame
    averages_df = pd.DataFrame([averages])

    # Convert to HTML
    html_table = averages_df.to_html(index=False, border=1)
    with open(output_file, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Last Minute Averages</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            {html_table}
        </body>
        </html>
        """)
    print(f"\t\tSaved last 1-minute averages to {output_file}.")
    print("\n-------------")
    for label, value in averages.items():
        print(label, "\t-\t", value)
    print("-------------\n")




def calculate_rolling_averages(data, time_spans):
    """Calculate rolling averages for each time span and save to CSV and HTML."""
    now = datetime.now(UTC)
    averages = {}

    for label, delta in time_spans.items():
        start_time = now - delta
        subset = data[data["Timestamp"] >= start_time]
        averages[label] = subset.mean(numeric_only=True)

    # Convert to DataFrame and save as CSV
    averages_df = pd.DataFrame(averages).T
    averages_df.index.name = "Time_Span"
    averages_df.to_csv(ROLLING_AVERAGES_FILE)
    print(f"\t\tSaved rolling averages to {ROLLING_AVERAGES_FILE}.")

    # Generate HTML table
    html_table = averages_df.to_html(border=1)
    html_file = ROLLING_AVERAGES_FILE.replace(".csv", ".html")
    with open(html_file, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rolling Averages</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Rolling Averages</h1>
            {html_table}
        </body>
        </html>
        """)
    print(f"\t\tSaved rolling averages HTML to {html_file}.")

    # Ensure the file is readable and writable by others
    os.chmod(ROLLING_AVERAGES_FILE, 0o664)
    os.chmod(html_file, 0o664)


def plot_system_metrics(csv_file_path, output_image_path):



    """
    Reads a CSV file containing system metrics, plots the data with dual y-axes, 
    and saves the plot as an image.

    Parameters:
    - csv_file_path: str, path to the input CSV file.
    - output_image_path: str, path to save the output plot image.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Convert the 'Timestamp' column to datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data.dropna(subset=["Timestamp"])  # Drop rows with invalid timestamps
    data = data.sort_values("Timestamp").reset_index(drop=True)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot CPU Temperature on the LHS axis
    ax1.plot(data['Timestamp'], data['CPU Temperature (°C)'], label='CPU Temperature (°C)', color='red', marker=',')
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("CPU Temperature (°C)", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a second y-axis for CPU and Memory Usage
    ax2 = ax1.twinx()
    ax2.plot(data['Timestamp'], data['CPU Usage (%)'], label='CPU Usage (%)', color='blue', marker=',')
    ax2.plot(data['Timestamp'], data['Memory Usage (%)'], label='Memory Usage (%)', color='green', marker=',')
    ax2.set_ylabel("Usage (%)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')



    # Plot CPU usage and temperature
    ax1.set_xlabel("Timestamp")
    ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    # Plot GPU temperature
    ax2.set_xlabel("Timestamp")
    ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))




    # Format the x-axis
    plt.xticks(rotation=45)
    plt.title("Weather Computer Metrics")
    plt.tight_layout()

    # Save the plot to the specified path
    plt.savefig(output_image_path)
    plt.close()


def parse_disk_usage(disk_usage_column):
    """Parses disk usage column to compute total used and total capacity."""
    total_used = 0
    total_capacity = 0
    for entry in disk_usage_column.split("; "):
        try:
            # Extract "used" and "total" values from the format
            parts = entry.split("(")[0].split(": ")[1].split("GB/")  # "33.00GB/68.35GB"
            used = float(parts[0])
            total = float(parts[1].split("GB")[0])
            total_used += used
            total_capacity += total
        except (IndexError, ValueError):
            continue  # Skip malformed entries
    return total_used, total_capacity


def clean_percentage(column):
    """Converts percentage strings like '50.9%' to float values."""
    return column.str.rstrip("%").astype(float)


def plot_system_stats(csv_file, output_image="system_stats_plot.png"):
    """Plots system stats from the CSV file and saves as an image."""
    # Load data
    df = pd.read_csv(csv_file, parse_dates=["Timestamp"])

    # Clean percentage columns
    for col in ["CPU Usage (%)", "Memory Usage (%)", "GPU Usage (%)", "GPU Memory Usage (%)"]:
        df[col] = clean_percentage(df[col])

    # Extract Disk Usage Data
    disk_totals = df["Disk Usage"].apply(parse_disk_usage)
    df["Disk Used (GB)"], df["Disk Total (GB)"] = zip(*disk_totals)

    # Extract Net Disk I/O
    net_io = df["Net Disk I/O (MB)"].str.extract(r"Read: ([\d.]+)MB, Write: ([\d.]+)MB").astype(float)
    df["Disk Read (MB)"], df["Disk Write (MB)"] = net_io[0], net_io[1]

    # Initialize subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle("System Stats", fontsize=18)

    time_formatter = DateFormatter("%H:%M")

    # Plot CPU usage and memory usage
    ax1 = axes[0, 0]
    ax1.plot(df["Timestamp"], df["CPU Usage (%)"], label="CPU Usage (%)", color="blue")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df["Timestamp"], df["Memory Usage (%)"], label="Memory Usage (%)", color="red")
    ax1.set_title("CPU and Memory Usage")
    ax1.set_ylabel("CPU Usage (%)")
    ax1_twin.set_ylabel("Memory Usage (%)")
    ax1.set_xlabel("Timestamp")
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.grid()

    # Plot CPU temperature
    ax2 = axes[0, 1]
    ax2.plot(df["Timestamp"], df["CPU Temp (°C)"].str.rstrip("°C").astype(float), label="CPU Temp (°C)", color="green")
    ax2.set_title("CPU Temperature")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_xlabel("Timestamp")
    ax2.legend(loc="upper left")
    ax2.grid()

    # Plot GPU usage and memory usage
    ax3 = axes[1, 0]
    ax3.plot(df["Timestamp"], df["GPU Usage (%)"], label="GPU Usage (%)", color="purple")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df["Timestamp"], df["GPU Memory Usage (%)"], label="GPU Memory Usage (%)", color="orange")
    ax3.set_title("GPU Usage and Memory")
    ax3.set_ylabel("GPU Usage (%)")
    ax3_twin.set_ylabel("GPU Memory Usage (%)")
    ax3.set_xlabel("Timestamp")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid()

    # Plot GPU temperature
    ax4 = axes[1, 1]
    ax4.plot(df["Timestamp"], df["GPU Temp (°C)"].str.rstrip("°C").astype(float), label="GPU Temp (°C)", color="brown")
    ax4.set_title("GPU Temperature")
    ax4.set_ylabel("Temperature (°C)")
    ax4.set_xlabel("Timestamp")
    ax4.legend(loc="upper left")
    ax4.grid()

    # Plot Disk Usage
    ax5 = axes[2, 0]
    ax5.plot(df["Timestamp"], df["Disk Used (GB)"], label="Disk Used (GB)", color="cyan")
    ax5.plot(df["Timestamp"], df["Disk Total (GB)"], label="Disk Total (GB)", color="black", linestyle="dashed")
    ax5.set_title("Disk Usage")
    ax5.set_ylabel("Disk Space (GB)")
    ax5.set_xlabel("Timestamp")
    ax5.legend(loc="upper left")
    ax5.grid()

    # Plot Net Disk I/O
    ax6 = axes[2, 1]
    ax6.plot(df["Timestamp"], df["Disk Read (MB)"], label="Disk Read (MB)", color="magenta")
    ax6.plot(df["Timestamp"], df["Disk Write (MB)"], label="Disk Write (MB)", color="orange")
    ax6.set_title("Net Disk I/O")
    ax6.set_ylabel("Data (MB)")
    ax6.set_xlabel("Timestamp")
    ax6.legend(loc="upper left")
    ax6.grid()

    # Plot other temperatures (Thermals)
    ax7 = axes[3, 0]

    # Ensure 'Thermals' column is treated as strings, handle NaN or invalid entries
    if "Thermals" in df.columns:
        df["Thermals"] = df["Thermals"].fillna("").astype(str)
        thermals = df["Thermals"].str.extractall(r"([\w\s]+)=([\d.]+)")  # Extract key-value pairs
        thermals = thermals.reset_index().pivot(index="level_0", columns=0, values=1)  # Reshape to wide format
        thermals = thermals.apply(pd.to_numeric, errors="coerce")  # Convert values to numeric
        for column in thermals.columns:
            ax7.plot(df["Timestamp"], thermals[column], label=column)
    else:
        ax7.text(0.5, 0.5, "No Thermal Data Available", ha="center", va="center", fontsize=12)

    ax7.set_title("Other Temperatures")
    ax7.set_ylabel("Temperature (°C)")
    ax7.set_xlabel("Timestamp")
    ax7.legend(loc="upper left")
    ax7.grid()

    # Adjust layout
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(time_formatter)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_image)
    plt.close()
    print(f"Plot saved as {output_image}")


def calculate_dew_point(temp_c, humidity, pressure_hpa):
    """
    Calculate the dew point temperature with pressure adjustment.
    """
    # Constants (adjusted for pressure):
    a = 17.625
    b = 243.04
    # Adjust saturation vapor pressure for the actual pressure
    alpha = (a * temp_c) / (b + temp_c) + np.log(humidity / 100)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point


def get_file_modification_times(directory="."):
    """
    Retrieves the modification times of files starting with 'weather_plot' 
    in the specified directory.
    
    Args:
        directory (str): The path to the directory. Defaults to the current directory.
        
    Returns:
        list of datetime: A list of modification times as datetime objects.
    """
    file_mod_times = []
    for filename in os.listdir(directory):
        if filename.startswith("weather_plot"):  # Check if the file name starts with 'weather_plot'
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):  # Ensure it's a file
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                file_mod_times.append(mod_time)
    return file_mod_times


def construct_time_spans(directory="."):
    """
    Constructs a dictionary of time spans based on the modification times of files.
    
    Args:
        directory (str): The path to the directory. Defaults to the current directory.
        
    Returns:
        dict: A dictionary of time spans to be included.
    """
    now = datetime.now()
    file_mod_times = get_file_modification_times(directory)
    time_spans = {}

    time_spans["10_minutes"] = timedelta(minutes=10)  # Always add 10 minutes

    if any((now - mod_time) > timedelta(hours=2) for mod_time in file_mod_times):
        time_spans["all_time"] = timedelta(days=3650)
    if any((now - mod_time) > timedelta(minutes=30) for mod_time in file_mod_times):
        time_spans["1_hour"] = timedelta(hours=1)
    if any((now - mod_time) > timedelta(hours=1) for mod_time in file_mod_times):
        time_spans["1_day"] = timedelta(days=1)
    if any((now - mod_time) > timedelta(days=1) for mod_time in file_mod_times):
        time_spans["1_year"] = timedelta(days=365),  # Effectively no limit    
        time_spans["1_month"] = timedelta(weeks=4),  # Effectively no limit                    

    return time_spans



def main():
    print("Starting server weather processing script...")

    local_stats_file = "my_pc_stats.csv"
    #initialize_csv(local_stats_file)
    gather_system_stats(local_stats_file)
    plot_system_stats("my_pc_stats.csv", "system_stats_plot.png")

    # Example usage
    time_spans = construct_time_spans()

    
    print("Starting new iteration!")        
    print("Reloading master file...")

    print("Making loss and system metric plots...")
    plot_system_metrics("system_usage.csv", "system_metrics_plot.png")


    print("Appending new data...")
    master_data = load_master_data(MASTER_FILE)
    master_data = append_new_data(master_data)
    master_data.loc[master_data["BMP_Temperature_C"] < 0, "DHT_Temperature_C"] *= -1

    forecaster = WeatherForecaster(master_file=MASTER_FILE, num_layers=6, batch_size=256, target_seq_length=3600)
    # Define the model file path
    model_path = "/media/bigdata/weather_station/weather_model.pth"

    # Check if the model file exists and is recent
    if os.path.exists(model_path):
        # Get the last modification time of the file
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        # Check if the file is older than a day
        if datetime.now() - file_mod_time > timedelta(days=1):
            print("Model file is older than a day. Retraining the model...")
            forecaster.train_model(epochs=300)
            forecaster.save_model(model_path)
            plot_training_loss("training_loss.csv", "training_loss_plot.png")
            plot_final_losses("final_losses.csv", "final_losses_plot.png")
        else:
            print("Model file is up-to-date. Loading the model...")
            forecaster.load_model(model_path)
    else:
        print("Model file does not exist. Training the model...")
        forecaster.train_model(epochs=300)
        forecaster.save_model(model_path)


    file_infer_time = datetime.fromtimestamp(os.path.getmtime(PREDICT_FILE))
    if datetime.now() - file_infer_time > timedelta(minutes=9):
        print("Running temp inference...")
        # Proceed with inference
        steps_ahead = 600
        recent_data = forecaster.load_master_data()
        timestamps = pd.to_datetime(recent_data["Timestamp"])  # Convert to datetime
        last_timestamp = timestamps.iloc[-1]  # The most recent timestamp
        data = recent_data[["DHT_Humidity_percent", "BMP_Temperature_C", "BMP_Pressure_hPa"]].values
        seq_length = forecaster.seq_length
        # Scale recent data using the training scale
        last_sequence = data[-forecaster.seq_length:]
        last_sequence[:, 1] = ((last_sequence[:, 1] - forecaster.temp_min) / (forecaster.temp_max - forecaster.temp_min))
        last_sequence[:, 0] = ((last_sequence[:, 0] - forecaster.hum_min) / (forecaster.hum_max - forecaster.hum_min))
        last_sequence[:, 2] = ((last_sequence[:, 2] - forecaster.pres_min) / (forecaster.pres_max - forecaster.pres_min))
        # Prepare recent sequence
        recent_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(forecaster.device)
        # Predict future temperatures
        predictions = forecaster.predict_future(recent_sequence, steps_ahead=steps_ahead)
        # Infer future timestamps
        interval_seconds = (timestamps.iloc[-1] - timestamps.iloc[-2]).total_seconds()
        future_timestamps = forecaster.infer_timestamps(last_timestamp, steps_ahead, interval_seconds)
        # Save predictions to CSV
        forecaster.save_predictions_to_csv(predictions, future_timestamps, PREDICT_FILE)



    predict_data = load_master_data(PREDICT_FILE)
    rolling_window = 10
    timestamps = np.arange(len(master_data["Timestamp"]))
    humidity = master_data["DHT_Humidity_percent"].values
    smooth_timestamps = np.linspace(0, len(timestamps) - 1, len(timestamps))
    
    print("Cleaning and smoothing data...")


    master_data["Sea_Level_Pressure_hPa"] = master_data["BMP_Pressure_hPa"] * (1 - (master_data["BMP_Altitude_m"] / 44330.77))**-5.255
    master_data["DHT_Temperature_Smoothed"] = master_data["DHT_Temperature_C"].rolling(window=rolling_window, min_periods=1, center=True).mean()
    master_data["BMP_Temperature_Smoothed"] = master_data["BMP_Temperature_C"].rolling(window=rolling_window, min_periods=1, center=True).mean()
    master_data["BMP_Pressure_hPa_Smoothed"] = master_data["BMP_Pressure_hPa"].rolling(window=rolling_window, min_periods=1, center=True).mean()        
    master_data["Sea_Level_Pressure_hPa_Smoothed"] = master_data["Sea_Level_Pressure_hPa"].rolling(window=rolling_window, min_periods=1, center=True).mean()                
    master_data["BH1750_Light_lx_Smoothed"] = master_data["BH1750_Light_lx"].rolling(window=rolling_window, min_periods=1, center=True).mean()                
    master_data["DHT_Humidity_percent_Smoothed"] = master_data["DHT_Humidity_percent"].rolling(window=rolling_window, min_periods=1, center=True).mean()                
    master_data["Median_Temperature_C"] = master_data[["BMP_Temperature_Smoothed", "DHT_Temperature_Smoothed"]].median(axis=1)
    master_data["Median_Temperature_F"] = master_data["Median_Temperature_C"] * 9 / 5 + 32
    master_data["Dew_Point_C"] = master_data.apply(lambda row: calculate_dew_point(row["Median_Temperature_C"], row["DHT_Humidity_percent"], row["BMP_Pressure_hPa"]),axis=1,)
    master_data["Dew_Point_C_smoothed"] = master_data["Dew_Point_C"].rolling(window=rolling_window, center=True).mean()                

    # Replace infinities with NaNs for numeric columns only
    numeric_cols = master_data.select_dtypes(include=[np.number]).columns
    master_data[numeric_cols] = master_data[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaNs in any column
    master_data.dropna(inplace=True)

    # Reset index for a clean DataFrame
    master_data.reset_index(drop=True, inplace=True)

    # Optional: Verify no remaining NaNs or infinities in numeric columns
    assert not master_data[numeric_cols].isnull().values.any(), "DataFrame still contains NaN values."
    assert not np.isinf(master_data[numeric_cols].values).any(), "DataFrame still contains infinite values."

    
    for label, delta in time_spans.items():
        print("\tGenerating plots...")
        subset = master_data[master_data["Timestamp"] >= datetime.now(UTC) - delta]
        generate_plots(subset, predict_data, f"/media/bigdata/weather_station/weather_plot_{label}.png", f"Weather Data ({label.replace('_', ' ').title()})")

    subset = master_data[master_data["Timestamp"] >= datetime.now(UTC) - timedelta(minutes=30)]
    generate_summary_plot(subset, f"/media/bigdata/weather_station/summary_plot.png")
    print("\t\tCalculating rolling averages...")
    calculate_rolling_averages(master_data, time_spans)
    save_last_minute_averages(master_data, predict_data, "/media/bigdata/weather_station/small_summary.html")
    print("Iteration complete!!!!! \n\n\tSleeping for 30 seconds...\n\n\n=============================================================================================")





if __name__ == "__main__":
    main()

