#!/usr/bin/env python3
"""
Alert System - Sends email notifications when plant station conditions require attention.

Currently implemented alerts:
1. High temperature alert (40°C or higher)

This script can be run manually and later automated.
"""

import os
import sys
import pandas as pd
import datetime
import smtplib
from email.mime.text import MIMEText

# --- Configuration ---
SMTP_USER = "cirrus.noreply@gmail.com"
SMTP_PASS = "jnlaisebvidlrioh"
EMAIL_TO = "niall.j.miller@gmail.com"

# File paths
PLANT_DATA_PATH = "/media/bigdata/plant_station/all_plant_data.csv"
ALERT_LOG_PATH = "/media/bigdata/plant_station/alerts.log"

# Alert thresholds
HIGH_TEMP_THRESHOLD = 40.0  # in Celsius

# --- Alert Functions ---

def check_high_temperature():
    """
    Check if plant station temperature exceeds threshold.
    Returns: (bool, str) - Alert triggered, Alert message
    """
    try:
        # Load the most recent data
        df = pd.read_csv(PLANT_DATA_PATH)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Get most recent readings
        recent_data = df.sort_values("Timestamp", ascending=False).head(5)
        max_temp = recent_data["Temperature_C"].max()
        
        if max_temp >= HIGH_TEMP_THRESHOLD:
            timestamp = recent_data.loc[recent_data["Temperature_C"].idxmax(), "Timestamp"]
            message = (f"HIGH TEMPERATURE ALERT: Plant station recorded {max_temp:.1f}°C "
                      f"at {timestamp}\n\n"
                      f"This exceeds the alert threshold of {HIGH_TEMP_THRESHOLD}°C.")
            return True, message
        return False, ""
    except Exception as e:
        return True, f"ERROR checking temperature: {str(e)}"

# Add more check functions here for future alert conditions
# def check_other_condition():
#     ...

def send_email(subject, body):
    """Send email alert."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def log_alert(alert_type, message, sent):
    """Log alert to file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SENT" if sent else "FAILED"
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(f"{timestamp} | {alert_type} | {status} | {message.splitlines()[0]}\n")

def main():
    """Main function to check conditions and send alerts."""
    print(f"Starting alert check at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for high temperature
    triggered, message = check_high_temperature()
    if triggered:
        print(f"HIGH TEMPERATURE ALERT: {message}")
        subject = "ALERT: Plant Station High Temperature"
        email_body = f"""
PLANT STATION ALERT
===================

{message}

This automatic alert was generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        sent = send_email(subject, email_body)
        log_alert("HIGH_TEMP", message, sent)
    else:
        print("Temperature normal, no alerts.")
    
    # Add more conditions here
    # e.g., if check_other_condition():
    #          ...
        
    print("Alert check completed.")

if __name__ == "__main__":
    main()
