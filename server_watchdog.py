import subprocess
import time

while True:
    try:
        subprocess.run(["python", "server_weather_processing.py"])
    except Exception as e:
        print(f"Script crashed: {e}. Restarting in 5 seconds...")
        time.sleep(5)

