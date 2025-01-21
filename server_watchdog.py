import subprocess
import time
import gc
import psutil
import os

def clean_ram():
    """Attempt to free up memory by running garbage collection."""
    gc.collect()
    print("Garbage collection completed.")

def kill_zombie_processes():
    """Terminate any zombie processes to release system resources."""
    for proc in psutil.process_iter(['pid', 'status']):
        if proc.info['status'] == psutil.STATUS_ZOMBIE:
            try:
                os.kill(proc.info['pid'], 9)
                print(f"Terminated zombie process with PID {proc.info['pid']}")
            except Exception as ex:
                print(f"Failed to kill process {proc.info['pid']}: {ex}")

while True:
    try:
        subprocess.run(["python", "server_weather_processing.py"], check=True)
        retry_time = 40
    except subprocess.CalledProcessError as e:
        retry_time = 5
        print(f"Script crashed with exit code {e.returncode}. Restarting in {retry_time} seconds...")
    except Exception as e:
        retry_time = 5
        print(f"Unexpected error: {e}. Restarting in {retry_time} seconds...")

    # Clean up RAM and kill zombie processes
    clean_ram()
    kill_zombie_processes()

    # Wait before restarting
    time.sleep(retry_time)

