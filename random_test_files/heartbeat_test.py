import datetime
import subprocess
import sys
import time


if __name__ == "__main__":
    looping = True
    heartbeat_process = subprocess.Popen(
        ['python3', '-c', '''
import time
import datetime
import sys

# Function to print the message every 5 minutes
def subprocess_print():
    while True:
        print(f"This subprocess is still running. Timestamp: {datetime.datetime.now()}", flush=True)
        time.sleep(10)

# Start the function
subprocess_print()
        '''],
        stdout=sys.stdout,  # Redirect stdout to the main process stdout
        stderr=sys.stderr,  # Optionally redirect stderr to the main process stderr
    )

    try:
        while looping:
            for i in range(5):
                print(f"The main process is still running. Timestamp: {datetime.datetime.now()}", flush=True)
                time.sleep(30)
            looping=False
            

    finally:
        heartbeat_process.terminate()