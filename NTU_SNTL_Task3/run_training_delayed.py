"""
Delayed training script
Waits 1 hour 30 minutes, then starts training with specified parameters
"""

import time
import subprocess
import sys
from datetime import datetime, timedelta

def main():
    # Wait time in seconds (1 hour 30 minutes = 5400 seconds)
    wait_seconds = 90 * 60  # 5400 seconds
    
    # Calculate start time
    start_time = datetime.now()
    target_time = start_time + timedelta(seconds=wait_seconds)
    
    print("=" * 70)
    print("DELAYED TRAINING SCRIPT")
    print("=" * 70)
    print(f"\nCurrent time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training will start at: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Waiting for {wait_seconds // 60} minutes ({wait_seconds} seconds)...")
    print("\nPress Ctrl+C to cancel")
    print("=" * 70)
    
    try:
        # Wait for specified time
        time.sleep(wait_seconds)
        
        print("\n" + "=" * 70)
        print("STARTING TRAINING NOW!")
        print("=" * 70)
        
        # Training command
        cmd = [
            sys.executable,  # Use current Python interpreter
            "main.py",
            "--project", "DCASE2025_Task3",
            "--exp", "MSI_FAFS_DN_Final",
            "--ms",
            "--iv",
            "--compfreq",
            "--dnorm",
            "--batch_size", "64",
            "--nb_epochs", "120",
            "--learning_rate", "0.001",
            "--weight_decay", "0.0001",
            "--dropout", "0.05",
            "--val_freq", "2"
        ]
        
        print(f"\nExecuting command:")
        print(" ".join(cmd))
        print("\n" + "=" * 70 + "\n")
        
        # Run training
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\nTraining cancelled by user.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n\nTraining failed with error code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
